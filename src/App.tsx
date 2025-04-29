import { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?worker";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import type { Table, StructRowProxy } from "apache-arrow";
import { TextEmbedder, TextEmbedderResult } from "@mediapipe/tasks-text";

const initDuckDB = async (
  setMyDuckDB: React.Dispatch<React.SetStateAction<duckdb.AsyncDuckDB | null>>,
  textEmbedder: TextEmbedder
) => {
  const worker = new duckdb_worker();
  const logger = new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);

  await db.instantiate(duckdb_wasm);
  await db.open({});

  const conn = await db.connect();
  await conn.query("LOAD vss;");
  await conn.query("INSTALL vss;");
  await conn.query("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;");
  await conn.query(
    "CREATE TABLE sora_doc (id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY, content VARCHAR);"
  );
  // @mediapipe/tasks-text の textEmbedder が作る Embedding はモデルによって変わるので要注意
  await conn.query("CREATE TABLE embeddings (vec FLOAT[512]);");

  const docs = [
    "こんにちは！ベクトル検索のデモです",
    "This is a demo of vector search",
    "今日はとても良い天気です。",
    "Today is a very nice day.",
    "こんにちは、世界！",
    "Hello, world!",
    "ベクトル検索って何ですか？",
    "What is vector search?",
    "ベクトル検索は、情報検索の一種で、データをベクトル空間にマッピングし、類似性を測定する手法です。",
  ];

  const placeholders = Array(512).fill("?").join(","); // 100次元の埋め込みベクトル
  const stmt2 = await conn.prepare(
    `INSERT INTO embeddings VALUES (ARRAY[${placeholders}]);`
  );

  for (const doc of docs) {
    const stmt = await conn.prepare(
      "INSERT INTO sora_doc (content) VALUES (?);"
    );
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

  setMyDuckDB(db);
};

const initEmbedding = async (
  setMyTextEmbedder: React.Dispatch<React.SetStateAction<any | null>>
) => {
  const text = await import("@mediapipe/tasks-text");
  const { TextEmbedder, FilesetResolver } = text;
  const textFiles = await FilesetResolver.forTextTasks(
    "../node_modules/@mediapipe/tasks-text/wasm"
  );
  const textEmbedder = await TextEmbedder.createFromOptions(textFiles, {
    baseOptions: {
      // modelAssetPath: `https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite`,
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite",
    },
  });
  setMyTextEmbedder(textEmbedder);
};

function App() {
  const [query, setQuery] = useState("こんにちは！ベクトル検索のデモです");
  const [queryEmbedding, setQueryEmbedding] =
    useState<TextEmbedderResult | null>(null);
  const [myTextEmbedder, setMyTextEmbedder] = useState<TextEmbedder | null>(
    null
  );
  const textEmbedderInitialized = useRef(false);
  const duckdbInitialized = useRef(false);
  const [myDuckDB, setMyDuckDB] = useState<duckdb.AsyncDuckDB | null>(null);
  const [resultRows, setResultRows] = useState<StructRowProxy<any>[]>([]);
  const debounceTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (textEmbedderInitialized.current) return;
    initEmbedding(setMyTextEmbedder).then(() => {
      textEmbedderInitialized.current = true;
      console.log("TextEmbedder initialized");
    });
  }, []);

  useEffect(() => {
    if (!duckdbInitialized.current && myTextEmbedder) {
      initDuckDB(setMyDuckDB, myTextEmbedder).then(() => {
        duckdbInitialized.current = true;
        console.log("DuckDB initialized");
      });
    }
  }, [myTextEmbedder]);

  // クエリをdebounceする処理
  const debouncedEmbedQuery = useCallback(
    (query: string) => {
      if (!myTextEmbedder) return;

      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = window.setTimeout(() => {
        console.log("Embedding query after debounce:", query);
        const newQueryEmbedding = myTextEmbedder.embed(query);
        setQueryEmbedding(newQueryEmbedding);
        console.log(
          "Query embedding:",
          newQueryEmbedding.embeddings[0].floatEmbedding?.length
        );
        debounceTimerRef.current = null;
      }, 800);
    },
    [myTextEmbedder]
  );

  useEffect(() => {
    if (!query) return;
    if (!myTextEmbedder) return;

    debouncedEmbedQuery(query);

    // クリーンアップ関数
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

      const conn = await myDuckDB.connect();

      const sql = `
      SELECT c.*, array_distance(e.vec, [${queryEmbedding.embeddings[0].floatEmbedding?.join(
        ","
      )}]::FLOAT[512]) AS distance
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
    };
    void doit();
  }, [query, myDuckDB, myTextEmbedder, queryEmbedding]);

  return (
    <div>
      <h1>DuckDB WASM VSS with MediaPipe - Demo App</h1>
      <p>
        This app demonstrates the integration of DuckDB VSS and MediaPipe
        text-task with React.
      </p>
      <div>
        <h2
          style={{
            textAlign: "left",
          }}
        >
          Query:
        </h2>
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
        {!duckdbInitialized.current || !textEmbedderInitialized.current ? (
          <div>
            <h2
              style={{
                textAlign: "left",
              }}
            >
              Initializing...
              <br />
              Text Embedder:{" "}
              {textEmbedderInitialized.current ? "OK" : "Loading..."}
              <br />
              DuckDB WASM: {duckdbInitialized.current ? "OK" : "Loading..."}
            </h2>
          </div>
        ) : null}
      </>
      <div>
        <h2
          style={{
            textAlign: "left",
          }}
        >
          Results:
        </h2>
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
