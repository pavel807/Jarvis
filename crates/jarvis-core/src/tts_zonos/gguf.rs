//! Чтение GGUF и загрузка весов (включая квантованные) в candle.
//!
//! Модели ZONOS3-Mini распространяются в GGUF: в файле лежат метаданные
//! (конфиг, токенизатор, silence-frame, границы бакетов условий) и веса.
//! Часть весов квантована: эксперты MoE лежат в Q4_0, миксеры (GDN
//! `in_proj`, внимание) — в Q8_0, остальное — F16/F32.
//!
//! Формат стандартный (GGUF v3 + блоки ggml), поэтому один загрузчик читает
//! и `backbone.gguf`, и `spk-encoder.gguf`, и `zac.gguf`.
//!
//! Раскладка тензоров в ggml: `ne[0]` — самый быстрый (непрерывный) размер,
//! данные идут row-major по `ne[0]`. Для линейных слоёв ggml хранит вес
//! транспонированным относительно PyTorch: `ne = [in, out]`, то есть форма
//! candle — `[out, in]`, ровно как ждёт `Linear`/`QMatMul`.

use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn as nn;
use candle_nn::Module;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// Тип значения в метаданных GGUF.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            Value::U8(v) => *v as i64,
            Value::I8(v) => *v as i64,
            Value::U16(v) => *v as i64,
            Value::I16(v) => *v as i64,
            Value::U32(v) => *v as i64,
            Value::I32(v) => *v as i64,
            Value::U64(v) => *v as i64,
            Value::I64(v) => *v,
            Value::Bool(v) => *v as i64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Value::F32(v) => *v as f64,
            Value::F64(v) => *v,
            _ => return self.as_i64().map(|v| v as f64),
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Описание одного тензора в файле.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// `ne[0]` — самый быстрый размер; в ggml это `ne0`.
    pub dims: Vec<usize>,
    pub ggml_type: u32,
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    /// Форма в «свете» candle: `ne` перевёрнуты, чтобы ne0 стал последним.
    pub fn candle_shape(&self) -> Vec<usize> {
        self.dims.iter().rev().copied().collect()
    }

    /// Число значений.
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Форма после снятия последнего измерения-стека (`ne`-последнего).
    pub fn per_item_candle_shape(&self) -> Vec<usize> {
        let mut shape = self.candle_shape();
        if shape.len() > 2 {
            shape.remove(0);
        }
        shape
    }
}

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_BF16: u32 = 30;

/// ggml-тип -> тип квантования candle.
pub fn ggml_dtype(ggml_type: u32) -> Option<GgmlDType> {
    Some(match ggml_type {
        GGML_TYPE_F32 => GgmlDType::F32,
        GGML_TYPE_F16 => GgmlDType::F16,
        GGML_TYPE_Q4_0 => GgmlDType::Q4_0,
        GGML_TYPE_Q4_1 => GgmlDType::Q4_1,
        GGML_TYPE_Q5_0 => GgmlDType::Q5_0,
        GGML_TYPE_Q5_1 => GgmlDType::Q5_1,
        GGML_TYPE_Q8_0 => GgmlDType::Q8_0,
        GGML_TYPE_Q8_1 => GgmlDType::Q8_1,
        GGML_TYPE_BF16 => GgmlDType::BF16,
        _ => return None,
    })
}

/// Прочитанный GGUF-файл: метаданные, описания тензоров и сами байты.
pub struct GgufFile {
    data: Vec<u8>,
    data_start: usize,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut data = Vec::new();
        File::open(path)
            .with_context(|| format!("Не удалось открыть GGUF: {:?}", path))?
            .read_to_end(&mut data)
            .with_context(|| format!("Не удалось прочитать GGUF: {:?}", path))?;
        Self::parse(data).with_context(|| format!("Разбор GGUF: {:?}", path))
    }

    fn parse(data: Vec<u8>) -> Result<Self> {
        let mut r = Reader {
            data: &data,
            pos: 0,
        };
        let magic = r.take(4)?;
        if &magic != b"GGUF" {
            bail!("Не GGUF-файл (magic={:?})", magic);
        }
        let version = r.u32()?;
        if version != 3 {
            bail!("Поддерживается только GGUF v3, получено v{version}");
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut kv = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty)?;
            kv.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()? as usize);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()? as usize;
            tensors.insert(
                name,
                TensorInfo {
                    dims,
                    ggml_type,
                    offset,
                    nbytes: 0,
                },
            );
        }

        // Данные начинаются с позиции, выровненной под general.alignment.
        let alignment = kv
            .get("general.alignment")
            .and_then(|v| v.as_i64())
            .unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(alignment) * alignment;

        let mut file = GgufFile {
            data,
            data_start,
            kv,
            tensors,
        };
        file.fix_sizes()?;
        Ok(file)
    }

    /// Проставляет реальные размеры тензоров (для квантов нужен тип блока).
    fn fix_sizes(&mut self) -> Result<()> {
        let infos: Vec<(String, TensorInfo)> = self
            .tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (name, mut info) in infos {
            let n = info.elem_count();
            let bytes = match info.ggml_type {
                GGML_TYPE_F32 => n * 4,
                GGML_TYPE_F16 | GGML_TYPE_BF16 => n * 2,
                GGML_TYPE_Q4_0 => n / 32 * 18,
                GGML_TYPE_Q4_1 => n / 32 * 20,
                GGML_TYPE_Q5_0 => n / 32 * 22,
                GGML_TYPE_Q5_1 => n / 32 * 24,
                GGML_TYPE_Q8_0 => n / 32 * 34,
                GGML_TYPE_Q8_1 => n / 32 * 36,
                GGML_TYPE_Q2_K => n / 256 * 84,
                GGML_TYPE_Q3_K => n / 256 * 110,
                GGML_TYPE_Q4_K => n / 256 * 144,
                GGML_TYPE_Q5_K => n / 256 * 176,
                GGML_TYPE_Q6_K => n / 256 * 210,
                GGML_TYPE_Q8_K => n / 256 * 292,
                other => bail!("Тензор «{name}»: неизвестный тип ggml {other}"),
            };
            info.nbytes = bytes;
            self.tensors.insert(name, info);
        }
        Ok(())
    }

    pub fn kv_str(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_str())
    }

    pub fn kv_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
    }

    pub fn kv_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_i64()).map(|v| v as u64)
    }

    pub fn kv_i64(&self, key: &str) -> Option<i64> {
        self.kv.get(key).and_then(|v| v.as_i64())
    }

    pub fn kv_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        self.kv.get(key).and_then(|v| v.as_array()).map(|items| {
            items
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect()
        })
    }

    pub fn kv_i64_array(&self, key: &str) -> Option<Vec<i64>> {
        self.kv.get(key).and_then(|v| v.as_array()).map(|items| {
            items
                .iter()
                .map(|v| v.as_i64().unwrap_or(0))
                .collect()
        })
    }

    pub fn kv_str_array(&self, key: &str) -> Option<Vec<String>> {
        self.kv.get(key).and_then(|v| v.as_array()).map(|items| {
            items
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("Нет тензора «{name}»"))
    }

    /// Сырые байты тензора (как в файле, с учётом выравнивания внутри шарда).
    fn raw(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        let start = self.data_start + info.offset;
        let end = start + info.nbytes;
        if end > self.data.len() {
            bail!(
                "Тензор «{name}» обрезан: нужно {end} байт, есть {}",
                self.data.len()
            );
        }
        Ok(&self.data[start..end])
    }

    /// Байты одного «элемента стека» (последнее измерение в терминах ne — первое
    /// в форме candle). Элементы лежат непрерывно, поэтому это просто срез.
    fn raw_item(&self, name: &str, index: usize) -> Result<&[u8]> {
        let info = self.info(name)?;
        let raw = self.raw(name)?;
        let dims = &info.dims;
        if dims.is_empty() {
            bail!("Тензор «{name}» без измерений");
        }
        let n_items = dims[dims.len() - 1];
        if index >= n_items {
            bail!("Тензор «{name}»: элемент {index} из {n_items}");
        }
        let per = info.nbytes / n_items;
        let start = index * per;
        Ok(&raw[start..start + per])
    }

    /// Тензор в f32 (деквантизация на лету). Форма — candle-совместимая.
    pub fn tensor_f32(&self, name: &str, device: &Device) -> Result<Tensor> {
        let info = self.info(name)?.clone();
        let shape = info.candle_shape();
        let values = self.dequantize(name)?;
        Tensor::from_vec(values, shape.as_slice(), device)
            .with_context(|| format!("Не удалось собрать тензор «{name}»"))
    }

    /// Тензор в f16 (вдвое меньше памяти).
    pub fn tensor_f16(&self, name: &str, device: &Device) -> Result<Tensor> {
        let info = self.info(name)?.clone();
        let shape = info.candle_shape();
        let values = self.dequantize(name)?;
        Tensor::from_vec(values, shape.as_slice(), device)?
            .to_dtype(DType::F16)
            .with_context(|| format!("Не удалось собрать f16-тензор «{name}»"))
    }

    /// Деквантизация в f32.
    pub fn dequantize(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?.clone();
        let raw = self.raw(name)?;
        let n = info.elem_count();
        match info.ggml_type {
            GGML_TYPE_F32 => Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            GGML_TYPE_F16 => Ok(raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            GGML_TYPE_BF16 => Ok(raw
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect()),
            GGML_TYPE_Q4_0 => {
                let mut out = Vec::with_capacity(n);
                for block in raw.chunks_exact(18) {
                    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    // ggml: qs[j] низкие нибблы — элементы j.., высокие — j+16..
                    for half in 0..2 {
                        for j in 0..16 {
                            let byte = block[2 + j];
                            let q = if half == 0 {
                                (byte & 0x0F) as i32 - 8
                            } else {
                                (byte >> 4) as i32 - 8
                            };
                            out.push(q as f32 * d);
                        }
                    }
                }
                Ok(out)
            }
            GGML_TYPE_Q8_0 => {
                let mut out = Vec::with_capacity(n);
                for block in raw.chunks_exact(34) {
                    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    for j in 0..32 {
                        out.push(block[2 + j] as i8 as f32 * d);
                    }
                }
                Ok(out)
            }
            other => bail!("Тензор «{name}»: квант {other} пока не декодируется"),
        }
    }

    /// Квантованный (или обычный) линейный вес: форма candle `[out, in]`.
    pub fn mat(&self, name: &str, device: &Device) -> Result<Mat> {
        let info = self.info(name)?.clone();
        let shape = info.candle_shape();
        if shape.len() != 2 {
            bail!("Вес «{name}»: ожидалась матрица, форма {shape:?}");
        }
        if let Some(dtype) = ggml_dtype(info.ggml_type) {
            let storage = QStorage::from_data(Cow::Borrowed(self.raw(name)?), device, dtype)
                .with_context(|| format!("QTensor «{name}»"))?;
            let qt = QTensor::new(storage, shape.clone())?;
            if matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
                let t = qt.dequantize(device)?;
                return Ok(Mat::Dense(nn::Linear::new(t, None)));
            }
            return Ok(Mat::Quant(QMatMul::from_qtensor(qt)?));
        }
        bail!("Вес «{name}»: тип ggml {} не поддержан", info.ggml_type)
    }

    /// Элемент стека как квантованный линейный вес.
    pub fn mat_item(&self, name: &str, index: usize, device: &Device) -> Result<Mat> {
        let info = self.info(name)?.clone();
        let shape = info.per_item_candle_shape();
        if shape.len() != 2 {
            bail!("Вес «{name}»: ожидалась стопка матриц, форма {shape:?}");
        }
        let dtype = ggml_dtype(info.ggml_type)
            .with_context(|| format!("Вес «{name}»: тип ggml {} не поддержан", info.ggml_type))?;
        let storage = QStorage::from_data(Cow::Borrowed(self.raw_item(name, index)?), device, dtype)
            .with_context(|| format!("QTensor «{name}»[{index}]"))?;
        let qt = QTensor::new(storage, shape.clone())?;
        if matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
            let t = qt.dequantize(device)?;
            return Ok(Mat::Dense(nn::Linear::new(t, None)));
        }
        Ok(Mat::Quant(QMatMul::from_qtensor(qt)?))
    }

    /// Число элементов стека (последнее измерение ne = первое в форме candle).
    pub fn stack_len(&self, name: &str) -> Result<usize> {
        let info = self.info(name)?;
        info.dims
            .last()
            .copied()
            .with_context(|| format!("Тензор «{name}» без измерений"))
    }
}

/// Линейный вес: обычный (f32/f16) или квантованный.
#[derive(Debug, Clone)]
pub enum Mat {
    Dense(nn::Linear),
    Quant(QMatMul),
}

impl Mat {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Mat::Dense(l) => Ok(l.forward(xs)?),
            Mat::Quant(q) => Ok(q.forward(xs)?),
        }
    }

    pub fn out_dim(&self) -> usize {
        match self {
            Mat::Dense(l) => l.weight().dim(0).unwrap_or(0),
            Mat::Quant(q) => match q {
                QMatMul::QTensor(t) => t.shape().dims()[0],
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dim(0).unwrap_or(0),
            },
        }
    }

    pub fn in_dim(&self) -> usize {
        match self {
            Mat::Dense(l) => l.weight().dim(1).unwrap_or(0),
            Mat::Quant(q) => match q {
                QMatMul::QTensor(t) => t.shape().dims()[1],
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dim(1).unwrap_or(0),
            },
        }
    }

    /// Развернуть в f32-тензор (для небольших весов, где нужен доступ к значениям).
    pub fn to_f32(&self, device: &Device) -> Result<Tensor> {
        match self {
            Mat::Dense(l) => Ok(l.weight().to_device(device)?.to_dtype(DType::F32)?),
            Mat::Quant(q) => match q {
                QMatMul::QTensor(t) => t.dequantize(device).map_err(Into::into),
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                    Ok(t.to_device(device)?.to_dtype(DType::F32)?)
                }
            },
        }
    }
}

/// Минимальный читатель GGUF.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("Переполнение в GGUF")?;
        if end > self.data.len() {
            bail!("GGUF обрезан на позиции {}", self.pos);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    items.push(self.value(elem)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            other => bail!("Неизвестный тип значения GGUF: {other}"),
        })
    }
}

/// f16 -> f32 (IEEE 754, half precision).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Субнормальное f16: значение = mant * 2^-24.
            let value = mant as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -value } else { value };
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> f16 с округлением к ближайшему (round-to-nearest-even).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // inf / nan
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }

    exp -= 127;
    if exp > 15 {
        return sign | 0x7C00; // overflow -> inf
    }
    if exp < -24 {
        return sign; // underflow -> zero
    }
    if exp < -14 {
        // субнормальное представление
        let shift = (-14 - exp) as u32;
        let full = mant | 0x800000;
        let mut out = (full >> (shift + 13)) as u16;
        let rem = full & ((1u32 << (shift + 13)) - 1);
        let half = 1u32 << (shift + 12);
        if rem > half || (rem == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out;
    }

    let mut out = ((exp + 15) as u16) << 10;
    let mut m = mant >> 13;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            let e = (out >> 10) + 1;
            let e = if e >= 0x1F { 0x7C00u16 } else { (e as u16) << 10 };
            out = e | m as u16;
            return sign | out;
        }
    }
    sign | out | m as u16
}

/// Dtype candle, соответствующий ggml-типу без квантования.
pub fn plain_dtype(ggml_type: u32) -> Option<DType> {
    match ggml_type {
        GGML_TYPE_F32 => Some(DType::F32),
        GGML_TYPE_F16 => Some(DType::F16),
        GGML_TYPE_BF16 => Some(DType::BF16),
        _ => None,
    }
}

/// Собирает квантованный тензор из сырых байт (для тестов и оракула).
pub fn qtensor_from_raw(
    raw: &[u8],
    shape: &[usize],
    dtype: GgmlDType,
    device: &Device,
) -> Result<Arc<QTensor>> {
    let storage = QStorage::from_data(Cow::Borrowed(raw), device, dtype)?;
    Ok(Arc::new(QTensor::new(storage, shape.to_vec())?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 1024.0, 3.14159, -0.0009765625] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() < v.abs() * 1e-3 + 1e-6, "{v} -> {back}");
        }
    }

    #[test]
    fn f16_specials() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C01).is_nan());
        assert_eq!(f16_to_f32(0x0001), 5.9604645e-8);
    }

    #[test]
    fn q4_0_layout() {
        // Один блок: d=1, qs[0]=0x21 → элемент 0 = 1, элемент 16 = 2.
        let mut raw = vec![0u8; 18];
        raw[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        raw[2] = 0x21;
        let file = GgufFile {
            data: raw.clone(),
            data_start: 0,
            kv: HashMap::new(),
            tensors: HashMap::from([(
                "t".to_string(),
                TensorInfo {
                    dims: vec![32],
                    ggml_type: GGML_TYPE_Q4_0,
                    offset: 0,
                    nbytes: 18,
                },
            )]),
        };
        let values = file.dequantize("t").unwrap();
        // ggml: значение = (ниббл - 8) * d. qs[0] = 0x21 -> младший 1 -> -7, старший 2 -> -6.
        assert_eq!(values[0], -7.0);
        assert_eq!(values[1], -8.0);
        assert_eq!(values[16], -6.0);
    }
}
