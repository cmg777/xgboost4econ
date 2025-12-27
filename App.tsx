
import React, { useState, useCallback, useEffect } from 'react';
import { AITutor } from './components/AITutor';
import { GlossaryTerm } from './components/GlossaryTerm';
import { Quiz } from './components/Quiz';
import * as Icons from './components/icons';
import { algorithmSteps, codeExamples, coreConceptsQuiz, algorithmQuiz } from './constants';
import { explainCode, generateUseCases, generateCaseStudyCode } from './services/geminiService';

const App: React.FC = () => {
    const [currentSection, setCurrentSection] = useState(0);
    const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({});
    const [showCode, setShowCode] = useState<Record<string, boolean>>({});
    const [copiedCodeId, setCopiedCodeId] = useState<string | null>(null);
    const [isDarkMode, setIsDarkMode] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('theme') === 'dark';
        }
        return false;
    });

    useEffect(() => {
        if (isDarkMode) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
    }, [isDarkMode]);

    const [geminiIsLoading, setGeminiIsLoading] = useState<Record<string, boolean>>({});
    const [geminiResults, setGeminiResults] = useState<Record<string, string>>({});
    const [geminiError, setGeminiError] = useState<string | null>(null);
    const [useCaseInput, setUseCaseInput] = useState("e-commerce");
    const [caseStudyInput, setCaseStudyInput] = useState("Predicting customer churn for a telecom company based on usage patterns and contract details.");

    const callGeminiAPI = useCallback(async (apiCall: () => Promise<string | null>, resultId: string) => {
        setGeminiIsLoading(prev => ({ ...prev, [resultId]: true }));
        setGeminiError(null);
        try {
            const text = await apiCall();
            if (text) {
                setGeminiResults(prev => ({ ...prev, [resultId]: text }));
            } else {
                throw new Error("Invalid response structure from API.");
            }
        } catch (error) {
            console.error("Gemini API call failed:", error);
            setGeminiError("Sorry, something went wrong while contacting the AI.");
            setGeminiResults(prev => ({ ...prev, [resultId]: '' }));
        } finally {
            setGeminiIsLoading(prev => ({ ...prev, [resultId]: false }));
        }
    }, []);

    const handleExplainCode = (code: string, title: string) => {
        const resultId = `explain_${title.replace(/\s+/g, '_')}`;
        callGeminiAPI(() => explainCode(code, title), resultId);
    };
    
    const handleGenerateUseCases = () => {
        callGeminiAPI(() => generateUseCases(useCaseInput), 'use_cases');
    };

    const handleGenerateCaseStudyCode = () => {
        callGeminiAPI(() => generateCaseStudyCode(caseStudyInput), 'case_study_code');
    };

    const toggleStep = (stepId: string) => setExpandedSteps(prev => ({ ...prev, [stepId]: !prev[stepId] }));
    const toggleCode = (codeId: string) => setShowCode(prev => ({ ...prev, [codeId]: !prev[codeId] }));
    
    const copyToClipboard = (text: string, id: string) => {
        navigator.clipboard.writeText(text).then(() => {
            setCopiedCodeId(id);
            setTimeout(() => setCopiedCodeId(null), 2000);
        }).catch(err => {
            console.error('Failed to copy: ', err);
        });
    };

    const sections = [
        { id: 0, title: "What is XGBoost?", icon: <Icons.Lightbulb className="w-5 h-5" />, content: (
            <div className="space-y-6">
                <div className="bg-sky-50 dark:bg-sky-900/20 p-6 rounded-lg border-l-4 border-sky-400"><h3 className="text-xl font-bold text-sky-900 dark:text-sky-200 mb-3">🚀 Extreme Gradient Boosting</h3><p className="text-slate-700 dark:text-slate-300 leading-relaxed">XGBoost is a powerful machine learning algorithm that builds a strong predictor by combining multiple <GlossaryTerm term="weak learners" definition="Simple models, typically small decision trees, that perform slightly better than random guessing." /> (usually decision trees). It's like a team of experts where each new expert learns from the mistakes of the previous ones!</p></div>
                <div className="grid md:grid-cols-2 gap-6"><div className="bg-emerald-50 dark:bg-emerald-900/20 p-5 rounded-lg border border-emerald-200 dark:border-emerald-800"><h4 className="font-bold text-emerald-900 dark:text-emerald-200 mb-3">✅ Key Advantages</h4><ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300 list-disc list-inside"><li>Excellent performance on tabular data</li><li>Fast training and prediction</li><li>Built-in <GlossaryTerm term="regularization" definition="A technique used to prevent overfitting by penalizing complex models. XGBoost uses L1 (Lasso) and L2 (Ridge) regularization." /> prevents overfitting</li><li>Handles missing values automatically</li><li>Works for both regression and classification</li></ul></div><div className="bg-violet-50 dark:bg-violet-900/20 p-5 rounded-lg border border-violet-200 dark:border-violet-800"><h4 className="font-bold text-violet-900 dark:text-violet-200 mb-3">🎯 Common Use Cases</h4><ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300 list-disc list-inside"><li>Financial risk assessment</li><li>E-commerce recommendation systems</li><li>Medical diagnosis prediction</li><li>Sales forecasting</li></ul></div></div>
                <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
                     <h3 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-3 flex items-center gap-2"><Icons.Sparkles className="w-6 h-6 text-violet-500"/> Brainstorm with Gemini AI</h3>
                     <p className="text-slate-600 dark:text-slate-400 mb-4">Enter an industry to see how XGBoost can be applied.</p>
                     <div className="flex flex-col sm:flex-row gap-2">
                         <input type="text" value={useCaseInput} onChange={(e) => setUseCaseInput(e.target.value)} className="w-full sm:w-1/2 p-2 border rounded-md bg-white dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-sky-500 focus:border-sky-500" placeholder="e.g., healthcare, finance"/>
                         <button onClick={handleGenerateUseCases} disabled={geminiIsLoading['use_cases']} className="flex items-center justify-center gap-2 px-4 py-2 bg-violet-600 text-white font-semibold rounded-md hover:bg-violet-700 disabled:bg-slate-400">
                             {geminiIsLoading['use_cases'] ? <><div className="spinner"></div><span>Generating...</span></> : <>✨ Generate Ideas</>}
                         </button>
                     </div>
                     {geminiResults['use_cases'] && <div className="gemini-content mt-4 p-4 bg-slate-50 dark:bg-slate-900/50 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300" dangerouslySetInnerHTML={{ __html: geminiResults['use_cases'] }}></div>}
                </div>
            </div>
        )},
        { id: 1, title: "Core Concepts", icon: <Icons.BookOpen className="w-5 h-5" />, content: (
            <div className="space-y-6">
                <div className="bg-amber-50 dark:bg-amber-900/20 p-5 rounded-lg border-l-4 border-amber-400"><h4 className="font-bold text-amber-900 dark:text-amber-200 mb-3">🌳 Gradient Boosting Basics</h4><p className="text-slate-700 dark:text-slate-300 mb-3">Imagine trying to predict house prices. Instead of building one complex model, you build many simple models sequentially, where each new model corrects the errors of the last. This process of sequential error-correction is the essence of <GlossaryTerm term="boosting" definition="An ensemble learning technique that builds a strong model by sequentially adding weak learners, where each new learner corrects the errors of its predecessor." />.</p></div>
                <div className="grid md:grid-cols-3 gap-4"><div className="bg-sky-50 dark:bg-sky-900/30 border border-sky-200 dark:border-sky-800 p-4 rounded-lg text-center"><Icons.TreePine className="w-8 h-8 text-sky-600 dark:text-sky-400 mx-auto mb-2" /><h5 className="font-bold text-sky-900 dark:text-sky-200">Weak Learners</h5><p className="text-sm text-slate-600 dark:text-slate-400">Simple decision trees.</p></div><div className="bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800 p-4 rounded-lg text-center"><Icons.BarChart3 className="w-8 h-8 text-emerald-600 dark:text-emerald-400 mx-auto mb-2" /><h5 className="font-bold text-emerald-900 dark:text-emerald-200">Gradient Descent</h5><p className="text-sm text-slate-600 dark:text-slate-400">An optimization algorithm to minimize errors.</p></div><div className="bg-violet-50 dark:bg-violet-900/30 border border-violet-200 dark:border-violet-800 p-4 rounded-lg text-center"><Icons.Play className="w-8 h-8 text-violet-600 dark:text-violet-400 mx-auto mb-2" /><h5 className="font-bold text-violet-900 dark:text-violet-200">Boosting</h5><p className="text-sm text-slate-600 dark:text-slate-400">Sequential error correction.</p></div></div>
                <Quiz questions={coreConceptsQuiz} />
            </div>
        )},
        { id: 2, title: "Algorithm Walkthrough", icon: <Icons.Play className="w-5 h-5" />, content: (
            <div className="space-y-4">
                {algorithmSteps.map((step) => (<div key={step.id} className="border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden"><button onClick={() => toggleStep(step.id)} className="w-full p-4 text-left flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"><span className="font-semibold text-slate-800 dark:text-slate-200">{step.title}</span>{expandedSteps[step.id] ? <Icons.ChevronDown className="text-sky-600" /> : <Icons.ChevronRight className="text-slate-500" />}</button>{expandedSteps[step.id] && (<div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-4 bg-white dark:bg-slate-800"><p className="text-slate-700 dark:text-slate-300">{step.detail}</p><div className="bg-slate-100 dark:bg-slate-700 p-4 rounded-lg"><div className="text-xs text-slate-500 dark:text-slate-400 font-medium mb-1">FORMULA</div><div className="font-mono text-sm text-slate-800 dark:text-slate-200">{step.formula}</div></div></div>)}</div>))}
                <div className="mt-8 bg-amber-50 dark:bg-amber-900/20 p-5 rounded-lg border-l-4 border-amber-400"><h4 className="font-bold text-amber-900 dark:text-amber-200 mb-2">🔑 Key Insight</h4><p className="text-slate-700 dark:text-slate-300">Each tree in XGBoost doesn't predict the target directly - it predicts the <GlossaryTerm term="residuals" definition="The error of the current model's prediction for each data point (Actual Value - Predicted Value). The next tree learns to predict these errors." /> (errors) from the previous stage. This sequential error correction is what makes boosting so powerful!</p></div>
                <Quiz questions={algorithmQuiz} />
            </div>
        )},
        { id: 3, title: "Python Implementation", icon: <Icons.Code className="w-5 h-5" />, content: (
            <div className="space-y-6">
                {codeExamples.map((example) => {
                    const resultId = `explain_${example.title.replace(/\s+/g, '_')}`;
                    return (
                        <div key={example.id} className="border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden">
                            <div className="bg-slate-50 dark:bg-slate-800/50 p-4"><h4 className="font-bold text-slate-800 dark:text-slate-200 mb-2">{example.title}</h4><p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{example.description}</p>
                                <div className="flex gap-2 flex-wrap">
                                    <button onClick={() => toggleCode(example.id)} className="flex items-center gap-2 px-3 py-1 bg-sky-600 text-white rounded-md text-sm hover:bg-sky-700"><Icons.Code className="w-4 h-4" />{showCode[example.id] ? 'Hide Code' : 'Show Code'}</button>
                                    <button onClick={() => handleExplainCode(example.code, example.title)} disabled={geminiIsLoading[resultId]} className="flex items-center justify-center gap-2 px-3 py-1 bg-violet-600 text-white rounded-md text-sm hover:bg-violet-700 disabled:bg-slate-400">
                                        {geminiIsLoading[resultId] ? <><div className="spinner"></div><span>Explaining...</span></> : <>✨ Explain Code</>}
                                    </button>
                                </div>
                            </div>
                            {geminiResults[resultId] && <div className="gemini-content p-4 border-t border-slate-200 dark:border-slate-700 bg-violet-50 dark:bg-violet-900/20 text-slate-700 dark:text-slate-300 text-sm" dangerouslySetInnerHTML={{ __html: geminiResults[resultId] }}></div>}
                            {showCode[example.id] && (<div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"><div className="flex justify-between items-center mb-2"><span className="text-sm text-slate-500 dark:text-slate-400 italic">Click to copy</span><button onClick={() => copyToClipboard(example.code, example.id)} className={`flex items-center gap-2 w-24 justify-center px-3 py-1 rounded-md text-sm text-white transition-all ${copiedCodeId === example.id ? 'bg-emerald-600' : 'bg-slate-700 hover:bg-slate-800 dark:bg-slate-600 dark:hover:bg-slate-500'}`}>{copiedCodeId === example.id ? 'Copied!' : <><Icons.Copy className="w-3 h-3" /> Copy</>}</button></div><pre className="bg-slate-900 text-sky-300 p-4 rounded-md text-sm overflow-x-auto"><code>{example.code}</code></pre></div>)}
                        </div>
                    );
                })}
                 <div key="case-study-generator" className="border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden mt-6 shadow-md">
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 text-white p-4">
                        <h4 className="font-bold text-lg mb-2 flex items-center gap-2"><Icons.Sparkles className="w-5 h-5 text-amber-300"/> 🤖 AI Case Study Generator</h4>
                        <p className="text-sm text-slate-300 mb-3">Describe a problem, and Gemini AI will generate a complete Python XGBoost solution, including advanced SHAP interpretation plots.</p>
                    </div>
                    <div className="p-4">
                        <textarea
                            value={caseStudyInput}
                            onChange={(e) => setCaseStudyInput(e.target.value)}
                            className="w-full p-2 border rounded-md text-sm h-24 bg-white dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                            placeholder="e.g., Predict house prices based on features like size and location."
                        ></textarea>
                        <button
                            onClick={handleGenerateCaseStudyCode}
                            disabled={geminiIsLoading['case_study_code']}
                            className="mt-2 flex items-center justify-center gap-2 w-full px-4 py-2 bg-slate-700 text-white font-semibold rounded-md hover:bg-slate-800 disabled:bg-slate-400"
                        >
                            {geminiIsLoading['case_study_code'] ? <><div className="spinner"></div><span>Generating Code...</span></> : <>Generate Python Code</>}
                        </button>
                    </div>
                    {geminiResults['case_study_code'] && (
                        <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-slate-600 dark:text-slate-300 font-semibold">Generated Solution</span>
                                <button onClick={() => copyToClipboard(geminiResults['case_study_code'], 'case_study')} className={`flex items-center gap-2 w-24 justify-center px-3 py-1 rounded-md text-sm text-white transition-all ${copiedCodeId === 'case_study' ? 'bg-emerald-600' : 'bg-slate-700 hover:bg-slate-800 dark:bg-slate-600 dark:hover:bg-slate-500'}`}>
                                    {copiedCodeId === 'case_study' ? 'Copied!' : <><Icons.Copy className="w-3 h-3" /> Copy</>}
                                </button>
                            </div>
                            <pre className="bg-slate-900 text-sky-300 p-4 rounded-md text-sm overflow-x-auto">
                                <code>{geminiResults['case_study_code']}</code>
                            </pre>
                        </div>
                    )}
                </div>
            </div>
        )},
        { id: 4, title: "Model Comparison", icon: <Icons.Scale className="w-5 h-5" />, content: (
            <div className="overflow-x-auto"><table className="w-full text-left border-collapse"><thead><tr><th className="p-4 bg-slate-100 dark:bg-slate-700 font-semibold border-b border-slate-200 dark:border-slate-600">Feature</th><th className="p-4 bg-sky-100 dark:bg-sky-900/50 font-semibold border-b border-sky-200 dark:border-sky-800 text-sky-900 dark:text-sky-200">XGBoost</th><th className="p-4 bg-slate-100 dark:bg-slate-700 font-semibold border-b border-slate-200 dark:border-slate-600">Random Forest</th><th className="p-4 bg-slate-100 dark:bg-slate-700 font-semibold border-b border-slate-200 dark:border-slate-600">Gradient Boosting</th></tr></thead><tbody className="text-sm"><tr className="hover:bg-slate-50 dark:hover:bg-slate-700"><td className="p-4 border-b border-slate-200 dark:border-slate-600 font-semibold text-slate-600 dark:text-slate-300">Model Building</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Sequential (trees correct previous errors)</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Parallel (independent trees vote)</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Sequential</td></tr><tr className="hover:bg-slate-50 dark:hover:bg-slate-700"><td className="p-4 border-b border-slate-200 dark:border-slate-600 font-semibold text-slate-600 dark:text-slate-300">Performance</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Typically highest predictive accuracy</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Strong, but often lower than boosting</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Very good, often close to XGBoost</td></tr><tr className="hover:bg-slate-50 dark:hover:bg-slate-700"><td className="p-4 border-b border-slate-200 dark:border-slate-600 font-semibold text-slate-600 dark:text-slate-300">Speed</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Highly optimized and parallelizable</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Fast to train (can be parallelized)</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Generally slower than XGBoost</td></tr><tr className="hover:bg-slate-50 dark:hover:bg-slate-700"><td className="p-4 border-b border-slate-200 dark:border-slate-600 font-semibold text-slate-600 dark:text-slate-300">Overfitting</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Controlled by regularization, early stopping, and tuning</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Less prone to overfitting due to bagging</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Can overfit without careful tuning</td></tr><tr className="hover:bg-slate-50 dark:hover:bg-slate-700"><td className="p-4 border-b border-slate-200 dark:border-slate-600 font-semibold text-slate-600 dark:text-slate-300">Key Feature</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Built-in regularization & optimized performance</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">Robustness and simplicity</td><td className="p-4 border-b border-slate-200 dark:border-slate-600">The foundational boosting algorithm</td></tr></tbody></table></div>
        )},
        { id: 5, title: "Common Pitfalls", icon: <Icons.AlertTriangle className="w-5 h-5" />, content: (
            <div className="space-y-6"><div className="bg-rose-50 dark:bg-rose-900/20 p-5 rounded-lg border-l-4 border-rose-400"><h4 className="font-bold text-rose-900 dark:text-rose-200 mb-2">🚨 Data Leakage</h4><p className="text-slate-700 dark:text-slate-300"><strong>Problem:</strong> Fitting a preprocessor (like a scaler) on the entire dataset before splitting. This "leaks" test set information into the training process.<br/><strong>Solution:</strong> Always split your data first. Fit preprocessors ONLY on the training data, then transform both train and test sets.</p></div><div className="bg-amber-50 dark:bg-amber-900/20 p-5 rounded-lg border-l-4 border-amber-400"><h4 className="font-bold text-amber-900 dark:text-amber-200 mb-2">🤔 Misinterpreting Feature Importance</h4><p className="text-slate-700 dark:text-slate-300"><strong>Problem:</strong> Default importance can be misleading. It shows which features a model *used*, not necessarily which are most *predictive*.<br/><strong>Solution:</strong> Use SHAP values for a more reliable understanding of feature contributions to predictions.</p></div><div className="bg-sky-50 dark:bg-sky-900/20 p-5 rounded-lg border-l-4 border-sky-400"><h4 className="font-bold text-sky-900 dark:text-sky-200 mb-2">🎯 Wrong Evaluation Metric</h4><p className="text-slate-700 dark:text-slate-300"><strong>Problem:</strong> Using "accuracy" on an imbalanced dataset (e.g., 99% non-fraud, 1% fraud) is misleading.<br/><strong>Solution:</strong> For imbalanced classification, use Precision-Recall, F1-Score, or AUC-ROC. For regression, consider RMSE vs. MAE.</p></div></div>
        )},
    ];
    
    return (
        <>
            <div className="max-w-6xl mx-auto p-4 sm:p-6 lg:p-8">
                <div className="relative text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 dark:text-slate-100 mb-2">XGBoost Learning Tool</h1>
                    <p className="text-lg text-slate-600 dark:text-slate-400">Enhanced with ✨ Gemini AI</p>
                    <button 
                        onClick={() => setIsDarkMode(!isDarkMode)}
                        className="absolute top-0 right-0 p-2 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                        aria-label="Toggle dark mode"
                    >
                        {isDarkMode ? <Icons.Sun className="w-6 h-6" /> : <Icons.Moon className="w-6 h-6" />}
                    </button>
                </div>
                {geminiError && <div className="max-w-4xl mx-auto mb-4 p-4 bg-rose-100 text-rose-700 border border-rose-300 rounded-md text-center">{geminiError}</div>}
                <div className="flex flex-wrap gap-2 mb-8 justify-center">{sections.map((section) => (<button key={section.id} onClick={() => setCurrentSection(section.id)} className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all text-sm sm:text-base ${currentSection === section.id ? 'bg-sky-600 text-white shadow-md' : 'bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-sky-100 dark:hover:bg-slate-600 hover:text-sky-700 dark:hover:text-sky-300 border border-slate-200 dark:border-slate-600'}`}>{section.icon}{section.title}</button>))}</div>
                <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-xl overflow-hidden">
                    <div className="p-6 md:p-8">
                        <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-8 flex items-center gap-3">
                            {React.cloneElement(sections[currentSection].icon, { className: 'w-8 h-8 text-sky-600 dark:text-sky-400' })}
                            {sections[currentSection].title}
                        </h2>
                        {sections[currentSection].content}
                    </div>
                </div>
                <div className="mt-12 bg-slate-100 dark:bg-slate-800/50 p-6 md:p-8 rounded-xl border border-slate-200 dark:border-slate-700">
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">📚 Quick Reference Guide</h3>
                    <div className="grid lg:grid-cols-2 gap-x-8 gap-y-10">
                        <div>
                            <h4 className="font-bold text-sky-800 dark:text-sky-300 mb-4">🎛️ Key Hyperparameters: The Control Knobs</h4>
                            <div className="space-y-4">
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-sky-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">n_estimators</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> The number of trees (or "experts") in your model.<br/><strong>Use:</strong> A higher number can improve accuracy but also increases training time and the risk of overfitting. Use with `early_stopping` to find the optimal number automatically.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-emerald-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">max_depth</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> The maximum complexity of each individual tree.<br/><strong>Use:</strong> Deeper trees can capture more complex patterns but are more likely to overfit. Start with a moderate value (e.g., 3-6) and tune from there.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-amber-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">learning_rate</code> (eta)</div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> How cautiously the model learns from each new tree's corrections.<br/><strong>Use:</strong> A small value (e.g., 0.01-0.2) makes the learning process more robust against overfitting but requires more `n_estimators`.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-violet-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">subsample</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> The fraction of data each tree gets to see.<br/><strong>Use:</strong> Setting this below 1.0 (e.g., 0.8) introduces randomness, which helps prevent overfitting by ensuring no single tree is biased by the full dataset.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-rose-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">colsample_bytree</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> The fraction of features (columns) each tree gets to use.<br/><strong>Use:</strong> Similar to `subsample`, this prevents the model from relying too heavily on a few strong features and improves generalization.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-yellow-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">reg_alpha/reg_lambda</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Intuition:</strong> Penalties to keep the model simple (L1 and L2 regularization).<br/><strong>Use:</strong> Increase these values if your model is overfitting. `reg_alpha` is good for high-dimensional data, while `reg_lambda` is a more general-purpose penalty.</p></div>
                            </div>
                        </div>
                        <div>
                            <h4 className="font-bold text-violet-800 dark:text-violet-300 mb-4">📊 Model Types: The Right Tool for the Job</h4>
                            <div className="space-y-4">
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-violet-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">XGBClassifier</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Use For:</strong> Predicting a category or label (e.g., "spam" vs. "not spam", "cat" vs. "dog").<br/><strong>Why:</strong> It's optimized for classification problems, where the output is a discrete class. It handles binary (two-class) and multi-class problems effectively.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-violet-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">XGBRegressor</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Use For:</strong> Predicting a continuous numerical value (e.g., house price, temperature).<br/><strong>Why:</strong> It's built to minimize the error between its numerical predictions and the actual numerical values, making it ideal for tasks where you need to estimate a quantity.</p></div>
                                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border-l-4 border-violet-400 shadow-sm"><div className="font-semibold text-slate-800 dark:text-slate-200"><code className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-1 rounded">XGBRanker</code></div><p className="text-sm text-slate-600 dark:text-slate-400 mt-1"><strong>Use For:</strong> Ranking a list of items by relevance (e.g., search results, product recommendations).<br/><strong>Why:</strong> It's specialized for learning-to-rank problems. It doesn't care about the exact prediction value, only about the relative order of the items, making it perfect for optimizing relevance.</p></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <AITutor />
        </>
    );
};

export default App;
