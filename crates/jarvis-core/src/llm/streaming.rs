//! Общие помощники потоковой генерации для обоих бэкендов (candle и uzu):
//! резка текста на куски для TTS и вырезание блоков рассуждений.

/// Сколько символов копим, прежде чем отдать кусок в TTS, если предложение
/// всё не кончается. Декод ~15-20 мс/токен, поэтому длинное первое
/// предложение откладывало бы звук.
pub(crate) const MAX_SENTENCE_CHARS: usize = 80;
/// С какого места можно разорвать длинную фразу по запятой/тире, чтобы
/// звук появился раньше.
pub(crate) const MIN_CLAUSE_CHARS: usize = 22;

pub(crate) fn find_sentence_end(buf: &str) -> Option<usize> {
    buf.find(['.', '!', '?', '\n']).map(|i| i + 1)
}

/// Граница куска для TTS. Сначала пробуем конец предложения; если оно
/// затянулось — отдаём часть по запятой, двоеточию или тире.
pub(crate) fn find_chunk_end(buf: &str) -> Option<usize> {
    if let Some(pos) = find_sentence_end(buf) {
        return Some(pos);
    }
    let mut count = 0usize;
    for (i, c) in buf.char_indices() {
        count += 1;
        if count >= MIN_CLAUSE_CHARS && matches!(c, ',' | ';' | ':' | '—' | '–') {
            return Some(i + c.len_utf8());
        }
    }
    None
}

pub(crate) fn speakable(sentence: &str) -> bool {
    sentence.chars().filter(|c| c.is_alphanumeric()).count() >= 2
}

pub(crate) fn strip_think(piece: &str, suppress: &mut bool) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut out = String::new();
    let mut rest = piece;

    loop {
        if *suppress {
            match rest.find(CLOSE) {
                Some(i) => {
                    rest = &rest[i + CLOSE.len()..];
                    *suppress = false;
                }
                None => break,
            }
        } else {
            match rest.find(OPEN) {
                Some(i) => {
                    out.push_str(&rest[..i]);
                    rest = &rest[i + OPEN.len()..];
                    *suppress = true;
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
    }

    out
}

/// Пушит дельту `delta` в буфер предложений и отдаёт готовые куски в TTS.
///
/// У uzu чанки стрима несут накопленный текст, поэтому дельту считаем
/// снаружи, а здесь — только резка и фильтрация.
pub(crate) fn push_delta(buf: &mut String, delta: &str, on_sentence: &mut dyn FnMut(&str)) {
    buf.push_str(delta);

    while let Some(pos) = find_chunk_end(buf) {
        let sentence: String = buf.drain(..pos).collect();
        if sentence.contains("<tool>") || sentence.contains("</tool>") {
            continue;
        }
        if speakable(&sentence) {
            on_sentence(&sentence);
        }
    }

    if buf.chars().count() > MAX_SENTENCE_CHARS
        && let Some(pos) = buf.rfind(' ')
    {
        let piece_flush: String = buf.drain(..pos + 1).collect();
        if !piece_flush.contains("<tool>")
            && !piece_flush.contains("</tool>")
            && speakable(&piece_flush)
        {
            on_sentence(&piece_flush);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_end_detection() {
        assert_eq!(find_sentence_end("Привет. Как дела"), Some(13));
        assert_eq!(find_sentence_end("Привет"), None);
        assert_eq!(find_sentence_end("Ваше\nВеличество"), Some(9));
    }

    #[test]
    fn chunks_split_at_clauses_for_early_speech() {
        assert_eq!(find_chunk_end("Привет"), None);
        assert_eq!(
            find_chunk_end("Привет, как дела?"),
            Some("Привет, как дела?".len())
        );
        let partial = "В Москве сегодня облачно, температура около минус трёх";
        let pos = find_chunk_end(partial).expect("должен найтись кусок");
        assert_eq!(&partial[..pos], "В Москве сегодня облачно,");
        let full = "В Москве сегодня облачно.";
        assert_eq!(find_chunk_end(full), Some(full.len()));
    }

    #[test]
    fn speakable_filter() {
        assert!(speakable("Стоп."));
        assert!(speakable("Да."));
        assert!(!speakable("..."));
        assert!(!speakable("—"));
    }

    #[test]
    fn think_block_stripping() {
        let mut suppress = false;
        assert_eq!(strip_think("Ответ.", &mut suppress), "Ответ.");
        assert_eq!(
            strip_think("<think>скрыто</think>Ответ.", &mut suppress),
            "Ответ."
        );

        let mut suppress = false;
        assert_eq!(strip_think("<think>часть", &mut suppress), "");
        assert!(suppress);
        assert_eq!(strip_think("1</think>Готово.", &mut suppress), "Готово.");
    }

    #[test]
    fn push_delta_emits_sentences() {
        let mut buf = String::new();
        let mut out = Vec::new();
        push_delta(&mut buf, "При", &mut |s| out.push(s.to_string()));
        assert!(out.is_empty());
        push_delta(&mut buf, "вет.", &mut |s| out.push(s.to_string()));
        assert_eq!(out, vec!["Привет.".to_string()]);
    }
}