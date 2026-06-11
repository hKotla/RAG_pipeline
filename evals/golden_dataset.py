"""
Hand-crafted Q&A pairs for the Climate Change in India corpus.

Sources indexed:
  - en.wikipedia.org/wiki/Climate_change_in_India
  - en.wikipedia.org/wiki/Climate_of_India
  - en.wikipedia.org/wiki/Renewable_energy_in_India
  - en.wikipedia.org/wiki/Deforestation_in_India

Each entry has:
  id                 -- unique identifier
  question           -- the query to run through the pipeline
  ground_truth_answer-- reference answer for relevance scoring
  relevant_keywords  -- words that should appear in relevant retrieved chunks
                        (used by retrieval_precision_at_k)
"""

GOLDEN_DATASET = [
    {
        "id": "q1",
        "question": "What are the main impacts of climate change on India's agriculture and food security?",
        "ground_truth_answer": (
            "Climate change threatens India's agriculture through reduced crop yields, "
            "water shortages from failing monsoons, and increased droughts. Over 56% of "
            "Indians work in agriculture, and 85% of farmers have low financial resilience, "
            "making them disproportionately vulnerable."
        ),
        "relevant_keywords": ["crop yield", "agriculture", "monsoon", "farmers", "food", "drought"],
    },
    {
        "id": "q2",
        "question": "How is India expanding its renewable energy capacity?",
        "ground_truth_answer": (
            "India aims to achieve 500 GW of non-fossil fuel capacity by 2030 and net-zero "
            "emissions by 2070. Solar power is growing at 113% year-over-year, and renewable "
            "sources now account for more than 50% of India's total installed electricity "
            "capacity. The Union budget allocated funds for the Green Energy Corridor."
        ),
        "relevant_keywords": ["renewable energy", "solar", "wind", "500 GW", "net zero", "electricity"],
    },
    {
        "id": "q3",
        "question": "How are Himalayan glaciers affected by climate change and what is the downstream impact on rivers?",
        "ground_truth_answer": (
            "Rising temperatures on the Tibetan Plateau are causing Himalayan glaciers to "
            "retreat, threatening the flow of the Ganges, Brahmaputra, Yamuna, and other "
            "major rivers, putting water supplies for hundreds of millions of people at risk."
        ),
        "relevant_keywords": ["glacier", "Himalayan", "Ganges", "Brahmaputra", "Tibetan Plateau", "river"],
    },
    {
        "id": "q4",
        "question": "What steps is India taking to reduce its greenhouse gas emissions?",
        "ground_truth_answer": (
            "India reduced its emission intensity of GDP by 24% between 2005 and 2016, "
            "targeting a one-third reduction by 2030. The country uses energy saving "
            "certificates and renewable energy certificates. India is a global leader in "
            "renewable energy with wind, solar, biomass, and hydro resources."
        ),
        "relevant_keywords": ["emission intensity", "greenhouse gas", "carbon", "NDC", "renewable", "net zero"],
    },
    {
        "id": "q5",
        "question": "What are the main causes of deforestation in India?",
        "ground_truth_answer": (
            "The main causes of deforestation in India include agricultural expansion, "
            "firewood collection, timber extraction, and colonization. These activities "
            "reduce India's carbon sinks and worsen climate change."
        ),
        "relevant_keywords": ["deforestation", "agricultural expansion", "firewood", "timber", "forest"],
    },
    {
        "id": "q6",
        "question": "How does India's industrial sector contribute to greenhouse gas emissions?",
        "ground_truth_answer": (
            "India's industrial sector — including cement, iron, and steel production — "
            "accounts for about a quarter of global industrial emissions. Fuel consumption "
            "surged 406% between 2000 and 2014. India is the world's second-largest steel "
            "producer, accounting for 7% of global crude steel production."
        ),
        "relevant_keywords": ["industrial", "cement", "steel", "iron", "emissions", "fuel consumption"],
    },
    {
        "id": "q7",
        "question": "What health effects does climate change have on India's population?",
        "ground_truth_answer": (
            "Air pollution and climate change are deeply intertwined in India. Extreme heat "
            "events threaten public health. Climate change also increases the risk of "
            "vector-borne diseases and worsens respiratory conditions from deteriorating "
            "air quality."
        ),
        "relevant_keywords": ["health", "air pollution", "heat", "disease", "respiratory"],
    },

    # --- Adversarial cases ---
    # These are designed to expose pipeline weaknesses.
    # A well-calibrated judge should return LOW scores here.

    {
        "id": "q8-out-of-corpus",
        "question": "What is the GDP of Bangladesh in 2024 and how does it compare to India?",
        "ground_truth_answer": "Not answerable from this corpus.",
        "relevant_keywords": ["Bangladesh", "GDP", "2024"],
        # Expected: Precision@K ~0.0 (no relevant chunks), Faithfulness ~1.0 (model says it doesn't know),
        #           Relevance ~0.0 or low (answer won't address the question)
    },
    {
        "id": "q9-unanswerable-states",
        "question": "Which specific Indian states have the highest flood risk and what are their adaptation budgets?",
        "ground_truth_answer": "Not answerable from this corpus — no state-level flood risk rankings or adaptation budgets are present.",
        "relevant_keywords": ["state", "flood risk", "adaptation budget", "Assam", "Kerala", "Bihar"],
        # Expected: Low Precision@K, low Relevance (answer will be vague or deflecting)
    },
    {
        "id": "q10-hallucination-bait",
        "question": "What did Prime Minister Modi announce at COP28 regarding India's coal phase-out timeline?",
        "ground_truth_answer": "Not answerable from this corpus.",
        "relevant_keywords": ["COP28", "Modi", "coal phase-out", "announcement"],
        # Expected: Low Precision@K, potentially low Faithfulness if model hallucinates COP28 details
    },
]
