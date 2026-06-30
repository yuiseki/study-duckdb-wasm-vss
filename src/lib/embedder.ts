// 埋め込みモデルの統一インターフェース。バックエンド(MediaPipe / Transformers.js)の差を吸収する。
// e5 系は "query: " / "passage: " の接頭辞が必要なため kind を受け取る。
export type Embedder = {
  dim: number;
  embed: (text: string, kind: "query" | "passage") => Promise<number[]>;
  close: () => void;
};

// Transformers.js バックエンドの Embedder を生成する。
// `@huggingface/transformers` は Node でもブラウザでも動くため、App.tsx と
// precompute スクリプトの両方からこの関数を共有して使う。
export const createTransformersEmbedder = async (
  modelId: string,
  dtype: string,
  pooling: "mean" | "cls",
  usePrefix: boolean
): Promise<Embedder> => {
  const { pipeline } = await import("@huggingface/transformers");
  // dtype を指定して量子化版 ONNX をロードする(例: q8 -> model_quantized.onnx)
  const extractor = await pipeline("feature-extraction", modelId, {
    dtype: dtype as never,
  });
  const prefix = (kind: "query" | "passage") =>
    usePrefix ? (kind === "query" ? "query: " : "passage: ") : "";
  const run = async (text: string, kind: "query" | "passage") => {
    // モデルの定義に合わせた pooling + 正規化で 1 文 1 ベクトルにする
    // (e5 系=mean, granite 系=cls)
    const out = await extractor(prefix(kind) + text, {
      pooling,
      normalize: true,
    });
    return Array.from(out.data as Float32Array).map((v) => Number(v));
  };
  const probe = await run("probe", "query");
  return {
    dim: probe.length,
    embed: run,
    close: () => {
      void (extractor as { dispose?: () => void }).dispose?.();
    },
  };
};
