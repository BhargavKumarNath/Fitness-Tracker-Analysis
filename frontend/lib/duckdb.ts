export interface FilteredActivitySummary { activityType: string; count: number; averageSteps: number; averageCalories: number; averageHeartRate: number; }

export async function queryActivitySummary(activityType: string): Promise<FilteredActivitySummary[]> {
  const duckdb = await import("@duckdb/duckdb-wasm/dist/duckdb-browser.mjs");
  const bundles = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(bundles);
  const workerUrl = URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker!}");`], { type: "text/javascript" }));
  const worker = new Worker(workerUrl);
  const database = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  await database.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(workerUrl);
  const fileUrl = `${window.location.origin}/data/fitness.parquet`;
  await database.registerFileURL("fitness.parquet", fileUrl, duckdb.DuckDBDataProtocol.HTTP, false);
  const connection = await database.connect();
  const escaped = activityType.replaceAll("'", "''");
  const result = await connection.query(`SELECT activity_type AS "activityType", COUNT(*) AS count, AVG(steps) AS "averageSteps", AVG(calories_burned) AS "averageCalories", AVG(heart_rate_avg) AS "averageHeartRate" FROM 'fitness.parquet' WHERE activity_type = '${escaped}' GROUP BY activity_type`);
  const rows = result.toArray() as unknown as Array<Record<string, string | number | bigint>>;
  await connection.close();
  await database.terminate();
  worker.terminate();
  return rows.map((row) => ({ activityType: String(row.activityType), count: Number(row.count), averageSteps: Number(row.averageSteps), averageCalories: Number(row.averageCalories), averageHeartRate: Number(row.averageHeartRate) }));
}