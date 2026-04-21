#!/usr/bin/env python3
"""Create quality gate prompts - tries LongBench, falls back to synthetic long-context prompts."""
import json, argparse, random

SYNTHETIC_PASSAGES = [
    "The history of artificial intelligence spans several decades. In the 1950s, Alan Turing proposed the Turing Test as a measure of machine intelligence. Early AI research focused on symbolic reasoning and expert systems. The development of neural networks in the 1980s marked a significant shift. The backpropagation algorithm enabled training of multi-layer networks. However, computational limitations constrained progress until the 2010s when deep learning revolutionized the field. Convolutional neural networks achieved superhuman performance on image recognition. Recurrent networks and LSTMs addressed sequence modeling. The introduction of the Transformer architecture in 2017 fundamentally changed natural language processing. Self-attention mechanisms allowed models to capture long-range dependencies. GPT and BERT demonstrated that pretraining on large corpora enabled transfer learning across diverse tasks.",
    "Climate change represents one of the most significant challenges facing humanity. The greenhouse effect occurs when atmospheric gases trap heat from the sun. Carbon dioxide, methane, and nitrous oxide are the primary greenhouse gases. Industrial activities since the nineteenth century have dramatically increased their concentrations. The Intergovernmental Panel on Climate Change has documented rising global temperatures. Arctic ice sheets are melting at unprecedented rates. Sea levels are rising globally, threatening coastal communities. Extreme weather events including hurricanes and droughts are becoming more frequent. Renewable energy sources such as solar and wind power offer pathways to decarbonization. International agreements like the Paris Accord aim to coordinate global responses. However, emissions reductions required to limit warming to 1.5 degrees Celsius remain challenging to achieve given current trajectories.",
    "The human immune system consists of multiple layers of defense against pathogens. The innate immune system provides immediate nonspecific responses to infection. Macrophages and neutrophils engulf and destroy invading microorganisms through phagocytosis. Natural killer cells eliminate virus-infected cells and tumor cells. The complement system marks pathogens for destruction through opsonization. The adaptive immune system provides specific long-lasting immunity. B lymphocytes produce antibodies that neutralize pathogens by binding to specific antigens. T lymphocytes coordinate immune responses and kill infected cells directly. Memory cells persist after infection and enable rapid responses upon re-exposure. Vaccines exploit this memory to confer immunity without natural infection. Autoimmune diseases arise when the immune system mistakenly attacks host tissues. Immunodeficiency disorders result from defects in immune system components.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales. Unlike classical physics, quantum systems exhibit wave-particle duality. The Heisenberg uncertainty principle states that position and momentum cannot both be precisely known simultaneously. Quantum superposition allows particles to exist in multiple states until measured. Entanglement creates correlations between distant particles that appear to violate locality. The Schrodinger equation governs the time evolution of quantum states. Quantum tunneling allows particles to pass through energy barriers classically forbidden to them. This phenomenon underlies nuclear fusion in stars and scanning tunneling microscopes. Quantum computers exploit superposition and entanglement to perform calculations exponentially faster than classical computers for certain problems. Quantum cryptography uses quantum mechanical properties to guarantee secure communication. The measurement problem remains one of the deepest unsolved questions in the foundations of physics.",
    "The Amazon rainforest is the largest tropical rainforest on Earth, covering approximately 5.5 million square kilometers. It spans nine countries in South America with Brazil containing about 60 percent of it. The Amazon basin contains approximately 390 billion individual trees representing 16,000 species. The forest generates 20 percent of the world's oxygen through photosynthesis. It harbors approximately 10 percent of all species on Earth including jaguars, anacondas, and thousands of bird species. The Amazon River discharges more fresh water into the ocean than any other river system. Indigenous communities have inhabited the Amazon for thousands of years and maintain deep ecological knowledge. Deforestation threatens the Amazon through agricultural expansion, cattle ranching, and logging operations. Scientists warn that the Amazon may be approaching a tipping point beyond which portions could convert to savanna. This would have catastrophic consequences for global climate regulation and biodiversity.",
]

QUESTIONS = [
    "What are the main themes discussed in this passage?",
    "Summarize the key points in two or three sentences.",
    "What are the most important facts mentioned?",
    "Describe the central argument or narrative.",
    "What challenges or problems are mentioned?",
]

def make_long_context_prompt(passage, question, repeat=3):
    context = " ".join([passage] * repeat)
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

def try_longbench(n=20):
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test", trust_remote_code=True)
        prompts = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            p = row.get("context", "") + "\n\n" + row.get("input", "")
            ans = row.get("answers", [""])[0] if row.get("answers") else ""
            prompts.append({"id": i, "prompt": p, "answer": ans})
        print(f"Loaded {len(prompts)} prompts from LongBench")
        return prompts
    except Exception as e:
        print(f"LongBench unavailable ({e}), using synthetic prompts")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic")
    args = ap.parse_args()

    prompts = None
    if not args.synthetic:
        prompts = try_longbench(args.n)

    if prompts is None:
        random.seed(42)
        prompts = []
        for i in range(args.n):
            passage = SYNTHETIC_PASSAGES[i % len(SYNTHETIC_PASSAGES)]
            question = QUESTIONS[i % len(QUESTIONS)]
            repeat = 3 + (i % 3)
            p = make_long_context_prompt(passage, question, repeat)
            prompts.append({"id": i, "prompt": p, "answer": ""})
        print(f"Created {len(prompts)} synthetic prompts")

    with open(args.out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.out}")

if __name__ == "__main__":
    main()
