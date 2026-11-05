/**
 * Минимальный Markdown → HTML для ответов ассистента.
 *
 * Без зависимостей: сначала экранируем HTML, потом превращаем разметку
 * в теги — пользовательский текст не может выполнить ни один тег.
 * Поддержано то, что реально приходит от локальной модели: заголовки,
 * списки, цитаты, код, инлайн-разметка, ссылки, разделители.
 */

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Инлайн-разметка поверх уже экранированного текста. */
function inline(s: string): string {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
    );
}

export function renderMarkdown(src: string): string {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out: string[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let quote: string[] = [];
  let code: string[] | null = null;

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${inline(esc(para.join(" ")))}</p>`);
      para = [];
    }
  };

  const flushList = () => {
    if (list) {
      const items = list.items.map((i) => `<li>${inline(esc(i))}</li>`).join("");
      out.push(list.ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`);
      list = null;
    }
  };

  const flushQuote = () => {
    if (quote.length) {
      out.push(`<blockquote>${inline(esc(quote.join(" ")))}</blockquote>`);
      quote = [];
    }
  };

  const flushAll = () => {
    flushPara();
    flushList();
    flushQuote();
  };

  for (const line of lines) {
    if (code !== null) {
      if (/^```/.test(line)) {
        out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
        code = null;
      } else {
        code.push(line);
      }
      continue;
    }

    if (/^```/.test(line)) {
      flushAll();
      code = [];
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)/);
    if (heading) {
      flushAll();
      const lvl = heading[1].length;
      out.push(`<h${lvl}>${inline(esc(heading[2].trim()))}</h${lvl}>`);
      continue;
    }

    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      flushAll();
      out.push("<hr>");
      continue;
    }

    const quoteLine = line.match(/^\s*>\s?(.*)/);
    if (quoteLine) {
      flushPara();
      flushList();
      quote.push(quoteLine[1]);
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ul || ol) {
      flushPara();
      flushQuote();
      const ordered = Boolean(ol);
      const item = (ul ? ul[1] : ol![1]).trim();
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(item);
      continue;
    }

    flushList();
    flushQuote();

    if (!line.trim()) {
      flushPara();
      continue;
    }
    para.push(line.trim());
  }

  if (code !== null) {
    // Незакрытый блок кода в стриминговом ответе — рисуем то, что успело прийти.
    out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  }
  flushAll();

  return out.join("");
}
