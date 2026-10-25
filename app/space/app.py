import torch
import sentencepiece as spm
import gradio as gr
from gpt import GPT, GPTConfig

sp = spm.SentencePieceProcessor(model_file="kam_bpe_8k.model")
ckpt = torch.load("gpt_best.pt", map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()


def continue_text(prompt, length, temperature, seed):
    torch.manual_seed(int(seed))
    ids = [sp.eos_id()] + sp.encode(prompt)
    idx = torch.tensor([ids])
    out = model.generate(idx, int(length), temperature=float(temperature),
                         top_k=50, stop_id=sp.eos_id())
    new = [t for t in out[0, 1:].tolist() if t != sp.eos_id()]
    return sp.decode(new)


demo = gr.Interface(
    fn=continue_text,
    inputs=[
        gr.Textbox(label="Start of a Kikamba sentence", value="Mũndũ"),
        gr.Slider(10, 200, value=80, step=10, label="Length (tokens)"),
        gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Number(value=0, label="Seed"),
    ],
    outputs=gr.Textbox(label="Kikamba GPT continues"),
    title="Kikamba GPT",
    description="A small GPT trained from scratch on Kikamba text. It continues whatever you start. It does not answer questions or translate.",
)

if __name__ == "__main__":
    demo.launch()