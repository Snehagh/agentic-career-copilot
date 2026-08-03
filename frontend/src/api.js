const BASE = "/api/v1";

export async function uploadFile(file, type) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload/${type}`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function analyze(jobTitle) {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_title: jobTitle }),
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.statusText}`);
  return res.json();
}
