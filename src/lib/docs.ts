// ベクトル検索の対象となる例文。日本語・英語をいろいろなトピックで混在させている。
// ブラウザ(App.tsx)と Node の precompute スクリプトで共有する単一ソース。
export const DOCS = [
  "こんにちは！ベクトル検索のデモです",
  "This is a demo of vector search",
  "今日はとても良い天気です。",
  "Today is a very nice day.",
  "こんにちは、世界！",
  "Hello, world!",
  "ベクトル検索って何ですか？",
  "What is vector search?",
  "ベクトル検索は、情報検索の一種で、データをベクトル空間にマッピングし、類似性を測定する手法です。",
  "明日は雨が降るみたいなので、傘を持って出かけましょう。",
  "It looks like it will rain tomorrow, so let's bring an umbrella.",
  "週末は家族と一緒に近くの公園でピクニックをしました。",
  "We had a picnic with our family at a nearby park over the weekend.",
  "猫はソファの上で気持ちよさそうに昼寝をしている。",
  "The cat is taking a comfortable nap on the sofa.",
  "新しいスマートフォンはカメラの性能が大幅に向上した。",
  "The new smartphone has a significantly improved camera.",
  "京都には歴史的な寺院や神社がたくさんあります。",
  "Kyoto has many historical temples and shrines.",
  "機械学習モデルの学習には大量のデータと計算資源が必要だ。",
  "Training a machine learning model requires a lot of data and compute.",
  "朝食にトーストとコーヒーを楽しむのが日課です。",
  "Enjoying toast and coffee for breakfast is my daily routine.",
  "電車が遅延したため、会議に少し遅刻してしまった。",
  "The train was delayed, so I was a little late for the meeting.",
  "この本はプログラミング初心者にとてもおすすめです。",
  "This book is highly recommended for programming beginners.",
  "海辺で夕日を眺めるのはとてもロマンチックだ。",
  "Watching the sunset by the sea is very romantic.",
  "健康のために毎朝30分のジョギングを続けている。",
  "I keep jogging for 30 minutes every morning to stay healthy.",
  "データベースのインデックスは検索を高速化するために使われる。",
  "Database indexes are used to speed up searches.",
];

// DOCS の内容から安定したハッシュを作る。例文を変更するとキャッシュ/parquet が自動的に無効化される。
export const DOCS_HASH = (() => {
  let h = 0;
  const s = DOCS.join("");
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
})();

// OPFS キャッシュキー・parquet ファイル名の共通キー。
// `<model-id>__<docs-hash>` という命名パターンで統一する。モデルを追加しても自動でこの規則に乗る。
export const cacheKeyFor = (modelId: string) => `${modelId}__${DOCS_HASH}`;

// 事前計算済み埋め込みベクトルの parquet ファイル名。
// 例: granite-97m-r2__abc123.parquet
export const parquetFileNameFor = (modelId: string) =>
  `${cacheKeyFor(modelId)}.parquet`;
