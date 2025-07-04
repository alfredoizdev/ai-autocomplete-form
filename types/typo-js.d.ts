declare module 'typo-js' {
  export default class Typo {
    constructor(lang: string, affData?: string, dicData?: string);
    check(word: string): boolean;
    suggest(word: string): string[];
  }
}