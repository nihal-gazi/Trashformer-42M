/**
 * GPT-2 Byte-Pair Encoding Tokenizer for Trashformer-42M
 */

function bytesToUnicode() {
  const bs = [];
  const cs = [];

  for (let b = 33; b <= 126; b++) { bs.push(b); cs.push(b); }
  for (let b = 161; b <= 172; b++) { bs.push(b); cs.push(b); }
  for (let b = 174; b <= 255; b++) { bs.push(b); cs.push(b); }

  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const d = {};
  for (let i = 0; i < bs.length; i++) {
    d[bs[i]] = String.fromCharCode(cs[i]);
  }
  return d;
}

const byteEncoder = bytesToUnicode();
const byteDecoder = {};
for (const [k, v] of Object.entries(byteEncoder)) {
  byteDecoder[v] = parseInt(k, 10);
}

export class BpeTokenizer {
  constructor(vocab, merges) {
    this.vocab = vocab;
    this.decoder = {};
    for (const [k, v] of Object.entries(vocab)) {
      this.decoder[v] = k;
    }

    this.bpeRanks = {};
    if (Array.isArray(merges)) {
      merges.forEach((m, idx) => {
        const pair = Array.isArray(m) ? m.join(' ') : m;
        this.bpeRanks[pair] = idx;
      });
    }

    this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  }

  getPairs(word) {
    const pairs = new Set();
    let prevChar = word[0];
    for (let i = 1; i < word.length; i++) {
      pairs.add(`${prevChar} ${word[i]}`);
      prevChar = word[i];
    }
    return pairs;
  }

  bpe(token) {
    let word = Array.from(token);
    let pairs = this.getPairs(word);
    if (pairs.size === 0) return [token];

    while (true) {
      let minRank = Infinity;
      let bestPair = null;

      for (const pair of pairs) {
        const rank = this.bpeRanks[pair];
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          bestPair = pair;
        }
      }

      if (bestPair === null) break;

      const [first, second] = bestPair.split(' ');
      const newWord = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
      pairs = this.getPairs(word);
    }

    return word;
  }

  encode(text) {
    if (!text) return [];
    const bpeTokens = [];
    const matches = text.match(this.pat) || [text];

    for (const token of matches) {
      const bytes = new TextEncoder().encode(token);
      let encodedToken = '';
      for (const b of bytes) {
        encodedToken += byteEncoder[b] || '';
      }

      for (const bpePart of this.bpe(encodedToken)) {
        if (this.vocab[bpePart] !== undefined) {
          bpeTokens.push(this.vocab[bpePart]);
        }
      }
    }
    return bpeTokens;
  }

  decode(tokens) {
    let text = '';
    for (const id of tokens) {
      if (this.decoder[id] !== undefined) {
        text += this.decoder[id];
      }
    }
    const bytes = [];
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (byteDecoder[ch] !== undefined) {
        bytes.push(byteDecoder[ch]);
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }

  decodeSingle(tokenId) {
    const raw = this.decoder[tokenId] || '';
    const bytes = [];
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if (byteDecoder[ch] !== undefined) {
        bytes.push(byteDecoder[ch]);
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }
}
