import { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?worker";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import type { Table, StructRowProxy } from "apache-arrow";
import { TextEmbedder, TextEmbedderResult } from "@mediapipe/tasks-text";

// MediaPipe の wasm は CDN から取得する。node_modules 相対パスはビルド後(本番)では解決できないため。
const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.35/wasm";

// MediaPipe text-embedder が公開しているモデル。サイズが大きいほど精度が高い傾向だが
// ダウンロード・初期化に時間がかかる。次元数はモデルによって異なるため実行時に判定する。
// 注意: MediaPipe の Average Word / Universal Sentence Encoder は語彙ベースの英語専用で、
// 空白で区切られない日本語文はすべて未知語となり同一ベクトルに潰れる（=検索が機能しない）。
// 日本語で意味のある結果が得られるのは文字レベルに分割できる BERT のみ。
const EMBEDDING_MODELS = [
  {
    id: "bert_embedder",
    label: "BERT（約26MB・日本語対応・推奨）",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite",
  },
  {
    id: "universal_sentence_encoder",
    label: "Universal Sentence Encoder（約6MB・英語のみ）",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite",
  },
  {
    id: "average_word_embedder",
    label: "Average Word Embedding（約0.7MB・最速・英語のみ）",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/average_word_embedder/float32/1/average_word_embedder.tflite",
  },
] as const;

const DEFAULT_MODEL_ID = "bert_embedder";

// ベクトル検索の対象となる例文。日本語・英語をいろいろなトピックで混在させている。
const DOCS = [
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

// 指定したモデルで TextEmbedder を生成する。
const createTextEmbedder = async (modelUrl: string): Promise<TextEmbedder> => {
  const { TextEmbedder, FilesetResolver } = await import(
    "@mediapipe/tasks-text"
  );
  const textFiles = await FilesetResolver.forTextTasks(MEDIAPIPE_WASM);
  return await TextEmbedder.createFromOptions(textFiles, {
    baseOptions: {
      modelAssetPath: modelUrl,
    },
  });
};

// textEmbedder で例文を埋め込み、DuckDB に VSS 用テーブルと HNSW インデックスを構築する。
// 埋め込み次元はモデルによって異なるため、最初の埋め込み結果から動的に決定する。
const buildDatabase = async (
  textEmbedder: TextEmbedder
): Promise<{ db: duckdb.AsyncDuckDB; dim: number }> => {
  const worker = new duckdb_worker();
  const logger = new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(duckdb_wasm);
  await db.open({});

  const conn = await db.connect();
  await conn.query("INSTALL vss;");
  await conn.query("LOAD vss;");

  // 埋め込み次元を判定する
  const probe = textEmbedder.embed("dimension probe");
  const dim = probe.embeddings[0].floatEmbedding?.length ?? 0;
  if (dim === 0) {
    throw new Error("埋め込み次元の判定に失敗しました");
  }

  await conn.query("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;");
  await conn.query(
    "CREATE TABLE sora_doc (id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY, content VARCHAR);"
  );
  await conn.query(`CREATE TABLE embeddings (vec FLOAT[${dim}]);`);

  const placeholders = Array(dim).fill("?").join(",");
  const stmt2 = await conn.prepare(
    `INSERT INTO embeddings VALUES (ARRAY[${placeholders}]);`
  );

  for (const doc of DOCS) {
    const stmt = await conn.prepare("INSERT INTO sora_doc (content) VALUES (?);");
    await stmt.query(doc);
    await stmt.close();
    const embedding = textEmbedder.embed(doc);
    const floatEmbedding = embedding.embeddings[0].floatEmbedding;
    if (floatEmbedding) {
      await stmt2.query(...floatEmbedding);
    }
  }
  await stmt2.close();

  await conn.query(`CREATE INDEX hnsw_index ON embeddings USING HNSW (vec);`);
  await conn.close();

  return { db, dim };
};

function App() {
  const [query, setQuery] = useState("こんにちは！ベクトル検索のデモです");
  const [selectedModelId, setSelectedModelId] =
    useState<string>(DEFAULT_MODEL_ID);
  const [queryEmbedding, setQueryEmbedding] =
    useState<TextEmbedderResult | null>(null);
  const [myTextEmbedder, setMyTextEmbedder] = useState<TextEmbedder | null>(
    null
  );
  const [myDuckDB, setMyDuckDB] = useState<duckdb.AsyncDuckDB | null>(null);
  const [embeddingDim, setEmbeddingDim] = useState<number | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading"
  );
  const [resultRows, setResultRows] = useState<StructRowProxy<any>[]>([]);
  const debounceTimerRef = useRef<number | null>(null);
  // 旧インスタンスのクリーンアップ用
  const embedderRef = useRef<TextEmbedder | null>(null);
  const dbRef = useRef<duckdb.AsyncDuckDB | null>(null);

  // モデル選択が変わるたびに、埋め込みモデルと DuckDB を作り直す。
  useEffect(() => {
    let cancelled = false;
    const rebuild = async () => {
      setStatus("loading");
      setResultRows([]);
      setQueryEmbedding(null);
      setMyTextEmbedder(null);
      setMyDuckDB(null);
      setEmbeddingDim(null);

      // 旧インスタンスを破棄
      embedderRef.current?.close();
      embedderRef.current = null;
      if (dbRef.current) {
        await dbRef.current.terminate();
        dbRef.current = null;
      }

      try {
        const model =
          EMBEDDING_MODELS.find((m) => m.id === selectedModelId) ??
          EMBEDDING_MODELS[0];
        const embedder = await createTextEmbedder(model.url);
        if (cancelled) {
          embedder.close();
          return;
        }
        embedderRef.current = embedder;

        const { db, dim } = await buildDatabase(embedder);
        if (cancelled) {
          await db.terminate();
          return;
        }
        dbRef.current = db;

        setMyTextEmbedder(embedder);
        setMyDuckDB(db);
        setEmbeddingDim(dim);
        setStatus("ready");
        console.log(`Model "${selectedModelId}" ready (dim=${dim})`);
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to initialize embedding model:", e);
          setStatus("error");
        }
      }
    };
    void rebuild();
    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  // クエリを debounce して埋め込む
  const debouncedEmbedQuery = useCallback(
    (query: string) => {
      if (!myTextEmbedder) return;

      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = window.setTimeout(() => {
        const newQueryEmbedding = myTextEmbedder.embed(query);
        setQueryEmbedding(newQueryEmbedding);
        debounceTimerRef.current = null;
      }, 800);
    },
    [myTextEmbedder]
  );

  useEffect(() => {
    if (!query) return;
    if (!myTextEmbedder) return;

    debouncedEmbedQuery(query);

    return () => {
      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query, myTextEmbedder, debouncedEmbedQuery]);

  useEffect(() => {
    const doit = async () => {
      if (!myDuckDB) return;
      if (!myTextEmbedder) return;
      if (!queryEmbedding) return;
      if (!embeddingDim) return;

      const conn = await myDuckDB.connect();

      const sql = `
      SELECT c.*, array_distance(e.vec, [${queryEmbedding.embeddings[0].floatEmbedding?.join(
        ","
      )}]::FLOAT[${embeddingDim}]) AS distance
      FROM embeddings e
      JOIN sora_doc c ON e.rowid = c.rowid
      ORDER BY distance ASC
      LIMIT 10;
    `;
      const newResults: Table = await conn.query(sql);
      const newResultRows: StructRowProxy<any>[] = newResults
        .toArray()
        .map((row: any) => JSON.parse(row));

      setResultRows(newResultRows);
      await conn.close();
    };
    void doit();
  }, [query, myDuckDB, myTextEmbedder, queryEmbedding, embeddingDim]);

  return (
    <div>
      <h1>DuckDB WASM VSS with MediaPipe - Demo App</h1>
      <p>
        This app demonstrates the integration of DuckDB VSS and MediaPipe
        text-task with React.
      </p>
      <div>
        <h2 style={{ textAlign: "left" }}>Embedding model:</h2>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            disabled={status === "loading"}
            style={{ padding: "4px" }}
          >
            {EMBEDDING_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {status === "ready" && embeddingDim ? (
            <span>次元数: {embeddingDim}</span>
          ) : null}
          {status === "loading" ? <span>モデル読み込み中...</span> : null}
          {status === "error" ? (
            <span style={{ color: "red" }}>モデルの読み込みに失敗しました</span>
          ) : null}
        </div>
      </div>
      <div>
        <h2 style={{ textAlign: "left" }}>Query:</h2>
        <div style={{ display: "flex", alignItems: "left" }}>
          <input
            style={{ width: "300px" }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your query here"
          />
        </div>
      </div>
      <>
        {status !== "ready" ? (
          <div>
            <h2 style={{ textAlign: "left" }}>
              {status === "loading"
                ? "Initializing... モデルとデータベースを準備しています"
                : "初期化に失敗しました"}
            </h2>
          </div>
        ) : null}
      </>
      <div>
        <h2 style={{ textAlign: "left" }}>Results:</h2>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            width: "100%",
            border: "1px solid #ddd",
            borderRadius: "4px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              display: "flex",
              background: "#f5f5f5",
              fontWeight: "bold",
              padding: "10px 0",
              borderBottom: "1px solid #ddd",
            }}
          >
            <div style={{ flex: "0 0 50px", padding: "0 10px" }}>ID</div>
            <div style={{ flex: "0 0 100px", padding: "0 10px" }}>Distance</div>
            <div style={{ flex: "1", padding: "0 10px" }}>Content</div>
          </div>
          {resultRows.map((row, index) => (
            <div
              key={index}
              style={{
                display: "flex",
                padding: "10px 0",
                borderBottom:
                  index < resultRows.length - 1 ? "1px solid #eee" : "none",
              }}
            >
              <div style={{ flex: "0 0 50px", padding: "0 10px" }}>
                {row.id}
              </div>
              <div style={{ flex: "0 0 100px", padding: "0 10px" }}>
                {row.distance?.toFixed(4)}
              </div>
              <div style={{ flex: "1", padding: "0 10px", textAlign: "left" }}>
                {row.content}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
