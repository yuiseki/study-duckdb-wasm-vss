// 埋め込みモデルの定義。ブラウザ(App.tsx)と Node の precompute スクリプトで共有する。
// バックエンド(MediaPipe / Transformers.js)の差を吸収するためのメタデータを持つ。
export type ModelConfig =
  | {
      id: string;
      label: string;
      backend: "mediapipe";
      url: string;
    }
  | {
      id: string;
      label: string;
      backend: "transformers";
      modelId: string;
      dtype: string;
      // プーリング方式。e5 系は mean、granite 系は cls
      pooling: "mean" | "cls";
      // e5 系は "query: " / "passage: " 接頭辞が必要。granite 系は不要
      usePrefix: boolean;
    };

// 注意: MediaPipe の Average Word / Universal Sentence Encoder は語彙ベースの英語専用で、
// 空白で区切られない日本語文はすべて未知語となり同一ベクトルに潰れる(=検索が機能しない)。
// 日本語で意味のある結果が得られるのは BERT(文字レベル分割) か Transformers.js の多言語モデル。
export const EMBEDDING_MODELS: ModelConfig[] = [
  // granite 系(IBM, Apache-2.0)。多言語(日本語含む)対応、CLS プーリング、接頭辞なし。
  // ONNX は yuiseki 自身が Apache-2.0 明示で再ホストしたもの。
  {
    id: "granite-97m-r2",
    label:
      "granite-embedding-97m-multilingual-r2（多言語・Apache-2.0・約93MB・推奨）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-97m-multilingual-r2-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-107m",
    label: "granite-embedding-107m-multilingual（多言語・Apache-2.0・約102MB）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-107m-multilingual-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-278m",
    label: "granite-embedding-278m-multilingual（多言語・Apache-2.0・約265MB）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-278m-multilingual-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-311m-r2",
    label:
      "granite-embedding-311m-multilingual-r2（多言語・Apache-2.0・約298MB・高精度）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-311m-multilingual-r2-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "multilingual-e5-base",
    label: "multilingual-e5-base（多言語・MIT・約265MB）",
    backend: "transformers",
    modelId: "onnx-community/multilingual-e5-base-ONNX",
    dtype: "q8",
    pooling: "mean",
    usePrefix: true,
  },
  {
    id: "bert_embedder",
    label: "MediaPipe BERT（約26MB・日本語可・速い）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite",
  },
  {
    id: "universal_sentence_encoder",
    label: "MediaPipe Universal Sentence Encoder（約6MB・英語のみ）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite",
  },
  {
    id: "average_word_embedder",
    label: "MediaPipe Average Word Embedding（約0.7MB・最速・英語のみ）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/average_word_embedder/float32/1/average_word_embedder.tflite",
  },
];

export const DEFAULT_MODEL_ID = "granite-97m-r2";

// 事前計算(precompute)で parquet を生成する対象モデルの id 一覧。
// 今は Transformers.js 系の granite-97m-r2 のみ。検証後に増やしていく。
// MediaPipe 系は Node での再現が困難なため現状は対象外。
export const PRECOMPUTE_MODEL_IDS = ["granite-97m-r2"];
