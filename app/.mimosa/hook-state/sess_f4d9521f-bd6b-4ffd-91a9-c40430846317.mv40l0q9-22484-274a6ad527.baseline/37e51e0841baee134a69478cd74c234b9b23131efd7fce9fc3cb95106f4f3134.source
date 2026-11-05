/**
 * Голоса Silero: бэкенд отдаёт технические id (`ru_dmitriy`) и пометку
 * «мужской»/«женский». Пользователю показываем имя и группируем по полу.
 */

const NAMES: Record<string, string> = {
  ru_eduard: "Эдуард",
  ru_dmitriy: "Дмитрий",
  ru_alexandr: "Александр",
  ru_bogdan: "Богдан",
  ru_igor: "Игорь",
  ru_marat: "Марат",
  ru_roman: "Роман",
  ru_gamat: "Гамат",
  ru_aigul: "Айгуль",
  ru_albina: "Альбина",
  ru_alfia: "Альфия",
  ru_alfia2: "Альфия II",
  ru_ekaterina: "Екатерина",
  ru_vika: "Вика",
  ru_karina: "Карина",
  ru_kermen: "Кермен",
  ru_miyau: "Мияу",
  ru_nurgul: "Нургуль",
  ru_oksana: "Оксана",
  ru_onaoy: "Оной",
  ru_ramilia: "Ррамиля",
  ru_safarhuja: "Сафархужа",
  ru_saida: "Саида",
  ru_sibday: "Сибдай",
  ru_zara: "Зара",
  ru_zhadyra: "Жадыра",
  ru_zhazira: "Жазира",
  ru_zinaida: "Зинаида",
  ru_kejilgan: "Кежилган",
};

export interface VoiceItem {
  id: string;
  name: string;
  note?: string;
}

export interface VoiceGroup {
  title: string;
  items: VoiceItem[];
}

export function voiceName(id: string): string {
  if (NAMES[id]) return NAMES[id];
  // Незнакомый голос — показываем id без служебного префикса.
  const bare = id.replace(/^ru_/, "").replace(/_/g, " ");
  return bare.charAt(0).toUpperCase() + bare.slice(1);
}

/** Разбивает список голосов на группы «Мужские» и «Женские». */
export function voiceGroups(voices: [string, string][]): VoiceGroup[] {
  const men: VoiceItem[] = [];
  const women: VoiceItem[] = [];
  const other: VoiceItem[] = [];

  for (const [id, desc] of voices) {
    const item: VoiceItem = { id, name: voiceName(id) };
    if (desc.includes("основн")) item.note = "основной";
    if (desc.includes("мужск")) men.push(item);
    else if (desc.includes("женск")) women.push(item);
    else other.push(item);
  }

  const groups: VoiceGroup[] = [
    { title: "Мужские", items: men },
    { title: "Женские", items: women },
    { title: "Другие", items: other },
  ];
  return groups.filter((g) => g.items.length > 0);
}
