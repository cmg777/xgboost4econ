
import React, { useState } from 'react';
import { QuizQuestion } from '../types';
import { CheckCircle, XCircle } from './icons';

interface QuizProps {
    questions: QuizQuestion[];
}

export const Quiz: React.FC<QuizProps> = ({ questions }) => {
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [selectedAnswers, setSelectedAnswers] = useState<Record<number, number>>({});
    const [showResults, setShowResults] = useState(false);
    
    const score = Object.keys(selectedAnswers).reduce((acc, key) => {
        const questionIndex = parseInt(key, 10);
        const question = questions[questionIndex];
        const selectedOptionIndex = selectedAnswers[questionIndex];
        return acc + (question.options[selectedOptionIndex].correct ? 1 : 0);
    }, 0);

    const handleAnswer = (optionIndex: number) => {
        setSelectedAnswers(prev => ({ ...prev, [currentQuestion]: optionIndex }));
    };

    const nextQuestion = () => {
        if (currentQuestion < questions.length - 1) {
            setCurrentQuestion(prev => prev + 1);
        } else {
            setShowResults(true);
        }
    };
    
    const restartQuiz = () => { 
        setCurrentQuestion(0); 
        setSelectedAnswers({}); 
        setShowResults(false); 
    };

    if (showResults) { 
        return (
            <div className="mt-8 bg-sky-50 dark:bg-sky-900/20 p-6 rounded-lg border-l-4 border-sky-400">
                <h4 className="text-xl font-bold text-sky-900 dark:text-sky-200 mb-4">Quiz Results!</h4>
                <p className="text-lg mb-4 text-slate-700 dark:text-slate-300">You scored {score} out of {questions.length}</p>
                <ul className="space-y-4">
                    {questions.map((q, index) => (
                        <li key={index} className="p-3 rounded-md bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600">
                            <p className="font-semibold text-slate-800 dark:text-slate-200">{q.question}</p>
                            <p className={`mt-2 text-sm ${q.options[selectedAnswers[index]]?.correct ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                                Your answer: {q.options[selectedAnswers[index]]?.text} 
                                {q.options[selectedAnswers[index]]?.correct ? <CheckCircle className="inline ml-2 w-4 h-4"/> : <XCircle className="inline ml-2 w-4 h-4"/>}
                            </p>
                            {!q.options[selectedAnswers[index]]?.correct && (
                                <p className="mt-1 text-sm text-emerald-700 dark:text-emerald-400">
                                    Correct answer: {q.options.find(o => o.correct)!.text}
                                </p>
                            )}
                        </li>
                    ))}
                </ul>
                <button onClick={restartQuiz} className="mt-6 px-4 py-2 bg-sky-600 text-white font-semibold rounded-md hover:bg-sky-700">Try Again</button>
            </div>
        ); 
    }
    
    const q = questions[currentQuestion];
    return (
        <div className="mt-8 bg-slate-50 dark:bg-slate-700/50 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
            <h4 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-2">🎓 Check Your Understanding</h4>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Question {currentQuestion + 1} of {questions.length}</p>
            <p className="font-semibold text-slate-700 dark:text-slate-300 mb-4">{q.question}</p>
            <div className="space-y-2 mb-6">
                {q.options.map((option, index) => (
                    <button 
                        key={index} 
                        onClick={() => handleAnswer(index)} 
                        className={`block w-full text-left p-3 rounded-md border-2 transition-all ${selectedAnswers[currentQuestion] === index ? 'border-sky-500 bg-sky-100 dark:bg-sky-900/50' : 'bg-white dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 border-slate-200 dark:border-slate-600'}`}
                    >
                        {option.text}
                    </button>
                ))}
            </div>
            <button 
                onClick={nextQuestion} 
                disabled={selectedAnswers[currentQuestion] === undefined} 
                className="px-5 py-2 bg-sky-600 text-white font-semibold rounded-md hover:bg-sky-700 disabled:bg-slate-300 dark:disabled:bg-slate-600"
            >
                {currentQuestion < questions.length - 1 ? 'Next Question' : 'Show Results'}
            </button>
        </div>
    );
};
