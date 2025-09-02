// api/hf.js
export default async function handler(req, res) {
  try {
    const response = await fetch(
      "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.HF_TOKEN}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ inputs: req.body.inputs })
      }
    );

    const data = await response.json();
    res.status(response.ok ? 200 : response.status).json(data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
}
