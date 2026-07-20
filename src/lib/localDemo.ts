import type { StoredHistoryItem } from "@/lib/historyStorage";
import type { AssessmentSession, BillingPlan, BillingStatus, CommunityAccount, CommunitySet, Reel } from "@/lib/types";

export type LocalDemoView = "account" | "player" | "quiz";

export const LOCAL_DEMO_AVAILABLE = process.env.NODE_ENV === "development";

export function isLocalDemoView(value: string | null, view: LocalDemoView): boolean {
  return LOCAL_DEMO_AVAILABLE && value === view;
}

export const LOCAL_DEMO_ACCOUNT: CommunityAccount = {
  id: "local-demo-pro",
  username: "reelai_demo",
  email: "demo@reelai.local",
  isVerified: true,
};

export const LOCAL_DEMO_BILLING_PLANS: BillingPlan[] = [
  { code: "free", name: "Free", monthly_price_cents: 0, daily_limit: 5 },
  { code: "plus", name: "Plus", monthly_price_cents: 499, daily_limit: 15 },
  { code: "pro", name: "Pro", monthly_price_cents: 1999, daily_limit: 50 },
];

export const LOCAL_DEMO_BILLING_STATUS: BillingStatus = {
  plan: "pro",
  daily_limit: 50,
  used_searches: 12,
  remaining_searches: 38,
  reset_at: new Date(Date.now() + 86_400_000).toISOString(),
  subscriptions: [
    {
      provider: "stripe",
      plan: "pro",
      status: "active",
      current_period_end: new Date(Date.now() + 30 * 86_400_000).toISOString(),
      cancel_at_period_end: false,
      product_id: "local-demo-pro",
    },
  ],
};

export const LOCAL_DEMO_HISTORY: StoredHistoryItem[] = [
  {
    materialId: "local-demo-learning-session",
    title: "Neural networks, visually explained",
    updatedAt: Date.now(),
    starred: false,
    generationMode: "slow",
    source: "search",
    feedQuery: "demo=player&return_tab=search",
    activeIndex: 0,
    activeReelId: "community:demo:neural-network",
  },
];

export const LOCAL_DEMO_REELS: Reel[] = [
  {
    reel_id: "community:demo:neural-network",
    material_id: "local-demo-learning-session",
    concept_id: "neural-network",
    concept_title: "Neural network structure",
    video_title: "But what is a neural network?",
    video_description: "A visual introduction to layers, activations, weights, and how a network turns pixels into a prediction.",
    channel_name: "3Blue1Brown",
    ai_summary: "A neural network transforms an input through layers of weighted activations. Training adjusts those weights so useful patterns produce stronger outputs.",
    video_id: "aircAruvnKk",
    video_url: "https://www.youtube.com/watch?v=aircAruvnKk",
    t_start: 66,
    t_end: 132,
    transcript_snippet: "Each neuron holds a number, and connections between layers determine how one pattern of activations produces the next.",
    takeaways: [
      "Layers progressively transform the representation.",
      "Weights control how strongly one activation affects another.",
      "Training changes weights to improve predictions.",
    ],
    captions: [
      { cue_id: "demo-nn-1", start: 0, end: 22, text: "Think of every neuron as holding a number between zero and one." },
      { cue_id: "demo-nn-2", start: 22, end: 46, text: "Those activations form a pattern that the next layer can interpret." },
    ],
    score: 0.98,
    relevance_score: 0.97,
    matched_terms: ["layers", "neurons", "weights"],
    relevance_reason: "Directly explains the core structure of a neural network.",
    match_reason: "Strong conceptual overview with a concrete visual model.",
    informativeness: 0.96,
    clip_duration_sec: 66,
    video_duration_sec: 1174,
    difficulty: 0.28,
    source_attribution: "Demo fixture · 3Blue1Brown on YouTube",
  },
  {
    reel_id: "community:demo:gradient-descent",
    material_id: "local-demo-learning-session",
    concept_id: "gradient-descent",
    concept_title: "Gradient descent",
    video_title: "Gradient descent, how neural networks learn",
    video_description: "An intuitive picture of a cost function and why following its downhill direction improves a model.",
    channel_name: "3Blue1Brown",
    ai_summary: "Gradient descent measures how the cost changes with each weight, then nudges every weight in the direction that reduces the total error.",
    video_id: "IHZwWFHWa-w",
    video_url: "https://www.youtube.com/watch?v=IHZwWFHWa-w",
    t_start: 88,
    t_end: 154,
    transcript_snippet: "The negative gradient tells you which direction decreases the cost most quickly from the model's current position.",
    takeaways: [
      "The cost function summarizes model error.",
      "The gradient points toward the steepest increase.",
      "Moving against it reduces error step by step.",
    ],
    captions: [
      { cue_id: "demo-gd-1", start: 0, end: 23, text: "Imagine the cost as the height of a landscape over all possible weights." },
      { cue_id: "demo-gd-2", start: 23, end: 50, text: "The gradient gives the direction of steepest ascent, so we move the other way." },
    ],
    score: 0.96,
    relevance_score: 0.95,
    matched_terms: ["gradient", "cost", "learning"],
    relevance_reason: "Connects the mathematical gradient to the model-training process.",
    match_reason: "The clip isolates the central intuition behind optimization.",
    informativeness: 0.94,
    clip_duration_sec: 66,
    video_duration_sec: 1280,
    difficulty: 0.46,
    source_attribution: "Demo fixture · 3Blue1Brown on YouTube",
  },
  {
    reel_id: "community:demo:calculus",
    material_id: "local-demo-learning-session",
    concept_id: "calculus",
    concept_title: "The derivative as change",
    video_title: "The essence of calculus",
    video_description: "A geometric explanation of how tiny changes reveal a function's instantaneous rate of change.",
    channel_name: "3Blue1Brown",
    ai_summary: "A derivative compares a tiny change in output with a tiny change in input. Shrinking the interval exposes the local rate of change.",
    video_id: "WUvTyaaNkzM",
    video_url: "https://www.youtube.com/watch?v=WUvTyaaNkzM",
    t_start: 104,
    t_end: 170,
    transcript_snippet: "Calculus becomes intuitive when a difficult shape is approximated by many tiny pieces whose behavior is easier to understand.",
    takeaways: [
      "Derivatives describe instantaneous change.",
      "Small approximations make nonlinear behavior manageable.",
      "The limiting idea connects geometry and algebra.",
    ],
    captions: [
      { cue_id: "demo-calc-1", start: 0, end: 24, text: "Break the change into pieces small enough that each one behaves almost linearly." },
      { cue_id: "demo-calc-2", start: 24, end: 50, text: "Then ask what happens as the size of those pieces approaches zero." },
    ],
    score: 0.94,
    relevance_score: 0.93,
    matched_terms: ["derivative", "change", "limit"],
    relevance_reason: "Introduces the visual intuition that unifies differential calculus.",
    match_reason: "A concise explanation of local linearity and limiting behavior.",
    informativeness: 0.93,
    clip_duration_sec: 66,
    video_duration_sec: 1020,
    difficulty: 0.38,
    source_attribution: "Demo fixture · 3Blue1Brown on YouTube",
  },
];

export const LOCAL_DEMO_ASSESSMENT_SESSION: AssessmentSession = {
  id: "local-demo-recall-check",
  material_id: "local-demo-learning-session",
  status: "pending",
  current_index: 0,
  question_count: 3,
  answered_count: 0,
  questions: [
    {
      id: "local-demo-question-network-weights",
      reel_id: "community:demo:neural-network",
      concept_id: "neural-network",
      concept_title: "Neural network structure",
      prompt: "What changes during training so a neural network can produce better predictions?",
      options: [
        "The number of input pixels",
        "The connection weights",
        "The video's frame rate",
        "The number of output labels",
      ],
    },
    {
      id: "local-demo-question-gradient-direction",
      reel_id: "community:demo:gradient-descent",
      concept_id: "gradient-descent",
      concept_title: "Gradient descent",
      prompt: "Why does gradient descent move in the direction opposite the gradient?",
      options: [
        "To make every weight equal",
        "To increase the cost as quickly as possible",
        "To reduce the cost from the model's current position",
        "To remove nonlinear activations",
      ],
    },
    {
      id: "local-demo-question-derivative",
      reel_id: "community:demo:calculus",
      concept_id: "calculus",
      concept_title: "The derivative as change",
      prompt: "What does a derivative describe at a particular point on a function?",
      options: [
        "Its instantaneous rate of change",
        "Its total area from zero",
        "Its largest possible output",
        "Its average value over all inputs",
      ],
    },
  ],
  score: null,
  understood_concepts: [],
  revisit_concepts: [],
  recent_accuracy: 0.75,
  rolling_accuracy: 0.72,
};

export const LOCAL_DEMO_ASSESSMENT_ANSWERS: Record<string, { correctIndex: number; explanation: string }> = {
  "local-demo-question-network-weights": {
    correctIndex: 1,
    explanation: "Training adjusts connection weights so useful activation patterns contribute more strongly to the prediction.",
  },
  "local-demo-question-gradient-direction": {
    correctIndex: 2,
    explanation: "The gradient points toward the steepest increase in cost, so moving against it decreases the model's error.",
  },
  "local-demo-question-derivative": {
    correctIndex: 0,
    explanation: "A derivative captures the local, instantaneous rate at which output changes as the input changes.",
  },
};

export const LOCAL_DEMO_COMMUNITY_SETS: CommunitySet[] = [
  {
    id: "local-demo-set-neural-networks",
    title: "Neural Networks: Visual Foundations",
    description: "A visual path through neurons, layers, weights, and the optimization loop that helps a model learn.",
    tags: ["machine learning", "neural networks", "visual learning"],
    reels: [
      {
        id: "local-demo-set-neural-networks-structure",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=aircAruvnKk",
        embedUrl: "https://www.youtube.com/embed/aircAruvnKk",
        tStartSec: 66,
        tEndSec: 132,
      },
      {
        id: "local-demo-set-neural-networks-training",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        embedUrl: "https://www.youtube.com/embed/IHZwWFHWa-w",
        tStartSec: 88,
        tEndSec: 154,
      },
    ],
    reelCount: 2,
    curator: "reelai_demo",
    likes: 18,
    learners: 42,
    updatedLabel: "Last Edited: today",
    updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(Date.now() - 4 * 86_400_000).toISOString(),
    thumbnailUrl: "/images/community/ai-systems.svg",
    featured: false,
  },
  {
    id: "local-demo-set-calculus-intuition",
    title: "Calculus Intuition Starter",
    description: "Build an intuitive picture of derivatives, local change, and why gradients power modern optimization.",
    tags: ["calculus", "derivatives", "intuition"],
    reels: [
      {
        id: "local-demo-set-calculus-intuition-derivative",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=WUvTyaaNkzM",
        embedUrl: "https://www.youtube.com/embed/WUvTyaaNkzM",
        tStartSec: 104,
        tEndSec: 170,
      },
      {
        id: "local-demo-set-calculus-intuition-gradient",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        embedUrl: "https://www.youtube.com/embed/IHZwWFHWa-w",
        tStartSec: 88,
        tEndSec: 154,
      },
    ],
    reelCount: 2,
    curator: "reelai_demo",
    likes: 11,
    learners: 29,
    updatedLabel: "Last Edited: yesterday",
    updatedAt: new Date(Date.now() - 26 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(Date.now() - 7 * 86_400_000).toISOString(),
    thumbnailUrl: "/images/community/calculus-flow.svg",
    featured: false,
  },
  {
    id: "local-demo-set-model-training-review",
    title: "Model Training Quick Review",
    description: "A compact review set for connecting network architecture, cost functions, and gradient descent before a quiz.",
    tags: ["machine learning", "review", "optimization"],
    reels: [
      {
        id: "local-demo-set-model-training-review-cost",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        embedUrl: "https://www.youtube.com/embed/IHZwWFHWa-w",
        tStartSec: 88,
        tEndSec: 154,
      },
      {
        id: "local-demo-set-model-training-review-network",
        platform: "youtube",
        sourceUrl: "https://www.youtube.com/watch?v=aircAruvnKk",
        embedUrl: "https://www.youtube.com/embed/aircAruvnKk",
        tStartSec: 66,
        tEndSec: 132,
      },
    ],
    reelCount: 2,
    curator: "reelai_demo",
    likes: 7,
    learners: 16,
    updatedLabel: "Last Edited: 3 days ago",
    updatedAt: new Date(Date.now() - 3 * 86_400_000).toISOString(),
    createdAt: new Date(Date.now() - 9 * 86_400_000).toISOString(),
    thumbnailUrl: "/images/community/physics-grid.svg",
    featured: false,
  },
];
