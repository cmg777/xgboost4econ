
import { GoogleGenAI } from "@google/genai";
import { ChatMessage } from "../types";

// Assume process.env.API_KEY is available in the execution environment as per the setup guidelines.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const callGemini = async (prompt: string): Promise<string | null> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-3-flash-preview',
            contents: prompt,
        });
        return response.text ?? null;
    } catch (error) {
        console.error("Error calling Gemini API:", error);
        throw error;
    }
}

export const explainCode = async (code: string, title: string): Promise<string | null> => {
    const prompt = `You are an expert data science tutor. Explain the following Python code snippet, titled "${title}", for a beginner. Go through it step-by-step, explaining what each part does and why it's important.

**IMPORTANT FORMATTING RULES:**
- Format your entire response using simple HTML tags. Do NOT use Markdown.
- Use <h3> for section titles (e.g., "Step 1: Imports").
- Use <p> for paragraphs of explanation.
- Use <strong> to bold key terms.
- For lists of points, use <ul> and <li>.
- Use <pre><code> to wrap entire code blocks for proper display.

Code:
\`\`\`python
${code}
\`\`\``;
    return callGemini(prompt);
};

export const generateUseCases = async (industry: string): Promise<string | null> => {
    const prompt = `You are a creative business strategist. Brainstorm 3 specific and innovative use cases for the XGBoost machine learning algorithm in the "${industry}" industry. For each use case, provide a brief explanation of the business problem and how XGBoost would provide a solution.

**IMPORTANT FORMATTING RULES:**
- Format your entire response using simple HTML tags. Do NOT use Markdown.
- Wrap the entire response in an <ol> tag.
- Each use case should be an <li> element.
- Inside each list item, use an <h4> for the use case title and <p> tags for the explanation.`;
    return callGemini(prompt);
};

export const generateCaseStudyCode = async (caseStudy: string): Promise<string | null> => {
    const prompt = `You are an expert data scientist and Python programmer. A user wants to solve a machine learning problem using XGBoost. Your task is to generate a single, complete, and runnable Python script for their case study.

**User's Case Study:**
"${caseStudy}"

**Requirements for the generated Python script:**
1.  **Runnable Code:** The script must be self-contained and runnable. To achieve this, you MUST use a suitable dataset from \`sklearn.datasets\` that plausibly matches the user's case study (e.g., \`load_breast_cancer\` for binary classification, \`fetch_california_housing\` for regression). Do NOT try to create a fake pandas DataFrame from scratch.
2.  **Best Practices:** Follow standard machine learning best practices, including:
    * Importing necessary libraries (\`xgboost\`, \`sklearn\`, \`pandas\`, \`numpy\`, \`shap\`, \`matplotlib.pyplot\`).
    * Loading and preparing the data (including creating a pandas DataFrame for easier use with SHAP).
    * Splitting the data into training and testing sets.
    * Training an appropriate XGBoost model (\`XGBClassifier\` or \`XGBRegressor\`).
    * Evaluating the model on the test set with appropriate metrics.
3.  **SHAP Interpretation (Crucial):** The script MUST include a comprehensive model interpretation section using the SHAP library.
    * Initialize the SHAP explainer (\`shap.TreeExplainer\`).
    * Calculate SHAP values.
    * **Global Interpretation:** Include code to generate and display AT LEAST TWO global SHAP plots (e.g., \`shap.summary_plot(shap_values, X_test)\` and \`shap.summary_plot(shap_values, X_test, plot_type="bar")\`).
    * **Local Interpretation:** Include code to explain a single prediction. Generate and display AT LEAST ONE local SHAP plot (e.g., \`shap.force_plot\` or \`shap.waterfall_plot\`).
4.  **Comments:** The code must be thoroughly commented to explain each step, especially the data preparation, model training, evaluation, and the meaning of each SHAP plot.
5.  **Output Format:** Provide ONLY the raw Python code. Do NOT wrap it in markdown backticks (\`\`\`) or add any explanatory text before or after the code.

Generate the Python script now.`;
    return callGemini(prompt);
};

export const getTutorResponse = async (chatHistory: ChatMessage[], newUserInput: string): Promise<string | null> => {
    const systemPrompt = `You are a friendly and encouraging AI tutor specializing in XGBoost and data science. Your goal is to help students understand the material presented in this learning app.
- Keep your answers concise, clear, and easy for a beginner to understand.
- Use analogies and simple examples.
- If the user asks a question unrelated to data science, machine learning, or XGBoost, gently guide them back to the topic.
- The user is interacting with you through a chat interface, so keep your responses conversational.
- Format your response using simple HTML tags like <p>, <strong>, <ul>, <ol>, and <li>. Do NOT use Markdown.`;
    
    const conversation = chatHistory.map(msg => `${msg.role}: ${msg.content}`).join('\n');
    const prompt = `${systemPrompt}\n\nHere is the conversation so far:\n${conversation}\n\nuser: ${newUserInput}\nassistant:`;
    
    return callGemini(prompt);
};
