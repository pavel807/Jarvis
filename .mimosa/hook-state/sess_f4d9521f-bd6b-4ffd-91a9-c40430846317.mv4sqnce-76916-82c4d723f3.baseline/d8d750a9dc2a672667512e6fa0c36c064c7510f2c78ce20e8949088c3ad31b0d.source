#!/bin/bash

# Создаем структуру папок для rustpotter
mkdir -p train/jarvis train/noise test/jarvis test/noise

# Функция для записи одного фрагмента
record_sample() {
    local folder=$1
    local prefix=$2
    local pitch_shift=$3 # Смещение голоса (в полутонах, например: -200, 0, 300)

    # Генерируем уникальное имя файла на основе таймстампа и случайного числа
    local timestamp=$(date +%s)_$((RANDOM % 1000))
    local temp_file="temp_${timestamp}.wav"
    local final_file="${folder}/${prefix}_${timestamp}.wav"

    echo "--------------------------------------------------------"
    read -p "Нажмите [ENTER] для НАЧАЛА записи..."

    # Запись в родном для macOS качестве (Mono, 16-bit), без флага -r 16000
    rec -q -c 1 -b 16 "$temp_file" &
    PID=$!

    read -p "ЗАПИСЬ ИДЕТ... Нажмите [ENTER] для ОСТАНОВКИ записи."
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null

    echo "Обработка аудио..."

    # Эффекты SoX:
    # 1. silence 1 0.1 1% - обрезает тишину в начале (дольше 0.1 сек с громкостью ниже 1%)
    # 2. reverse + silence + reverse - обрезает тишину в конце файла
    # 3. pitch X - изменяет высоту голоса (X в центах, 100 центов = 1 полутон)
    # 4. norm - нормализует громкость, чтобы все файлы звучали одинаково хорошо

        if [ "$pitch_shift" -eq 0 ]; then
            # Было: ... reverse norm
            sox "$temp_file" "$final_file" silence 1 0.1 1% reverse silence 1 0.1 1% reverse norm -1
        else
            # Было: ... pitch $pitch_shift norm
            sox "$temp_file" "$final_file" silence 1 0.1 1% reverse silence 1 0.1 1% reverse pitch $pitch_shift norm -1
        fi

    # Удаляем временный файл
    rm -f "$temp_file"
    echo "Сохранено: $final_file"
}

# Главный цикл интерфейса
while true; do
    clear
    echo "=== ЗАПИСЬ ДАТАСЕТА ДЛЯ RUSTPOTTER ==="
    echo "1) Записать 'Jarvis' для ОБУЧЕНИЯ (Train)"
    echo "2) Записать 'Noise' для ОБУЧЕНИЯ (Train)"
    echo "3) Записать 'Jarvis' для ТЕСТА (Test)"
    echo "4) Записать 'Noise' для ТЕСТА (Test)"
    echo "5) Выход"
    read -p "Выберите опцию (1-5): " opt

    case $opt in
        1)
            echo "Запись слова 'Jarvis' (Train). Сделаем 3 варианта: свой, робот (низкий), писклявый (высокий)."
            record_sample "train/jarvis" "normal" 0
            record_sample "train/jarvis" "low_pitch" -300
            record_sample "train/jarvis" "high_pitch" 400
            ;;
        2)
            echo "Запись шума/тишины (Train). Просто молчите или шумите во время записи."
            record_sample "train/noise" "noise" 0
            ;;
        3)
            echo "Запись слова 'Jarvis' (Test)."
            record_sample "test/jarvis" "normal" 0
            record_sample "test/jarvis" "pitch" 200
            ;;
        4)
            echo "Запись шума/тишины (Test)."
            record_sample "test/noise" "noise" 0
            ;;
        5)
            echo "Выход. Удачи в обучении rustpotter!"
            exit 0
            ;;
        *)
            echo "Неверный выбор, попробуйте еще раз."
            sleep 1
            ;;
    esac
    echo -e "\nГотово! Нажмите Enter для возврата в меню."
    read
done
