
export interface QuizQuestionOption {
    text: string;
    correct: boolean;
}

export interface QuizQuestion {
    question: string;
    options: QuizQuestionOption[];
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface CodeExample {
    id: string;
    title: string;
    description: string;
    code: string;
}

export interface AlgorithmStep {
    id:string;
    title: string;
    detail: string;
    formula: string;
    code?: string;
}