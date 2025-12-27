
import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../types';
import { getTutorResponse } from '../services/geminiService';
import { X, Send, MessageCircle } from './icons';

export const AITutor: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
        { role: 'assistant', content: "Hello! I'm your AI tutor. Ask me anything about XGBoost or the concepts in this app!" }
    ]);
    const [userInput, setUserInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatHistory]);

    const handleSendMessage = async () => {
        if (!userInput.trim() || isLoading) return;
        
        const newUserMessage: ChatMessage = { role: 'user', content: userInput };
        setChatHistory(prev => [...prev, newUserMessage]);
        setUserInput("");
        setIsLoading(true);

        try {
            const assistantResponse = await getTutorResponse(chatHistory, userInput);
            const newAssistantMessage: ChatMessage = { 
                role: 'assistant', 
                content: assistantResponse || "I'm not sure how to answer that. Could you rephrase?" 
            };
            setChatHistory(prev => [...prev, newAssistantMessage]);
        } catch (error) {
            console.error("AI Tutor API call failed:", error);
            const errorMessage: ChatMessage = { 
                role: 'assistant', 
                content: "Sorry, I'm having a little trouble connecting. Please try again in a moment." 
            };
            setChatHistory(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed bottom-5 right-5 z-50">
            <div className={`transition-all duration-300 ${isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}`}>
                <div className="w-[calc(100vw-40px)] sm:w-96 h-[60vh] bg-white dark:bg-slate-800 rounded-xl shadow-2xl flex flex-col border border-slate-200 dark:border-slate-700">
                   <div className="flex justify-between items-center p-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700 rounded-t-xl">
                        <h3 className="font-bold text-slate-800 dark:text-slate-200">✨ AI Tutor</h3>
                        <button onClick={() => setIsOpen(false)} className="text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100"><X className="w-5 h-5"/></button>
                   </div>
                    <div className="flex-1 p-4 overflow-y-auto tutor-chat-history bg-white dark:bg-slate-800">
                        <div className="space-y-4">
                            {chatHistory.map((msg, index) => (
                                <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-sky-600 text-white' : 'bg-slate-200 text-slate-800 dark:bg-slate-600 dark:text-slate-200'}`}>
                                        <div className="gemini-content text-sm" dangerouslySetInnerHTML={{ __html: msg.content }}></div>
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="max-w-[80%] p-3 rounded-lg bg-slate-200 dark:bg-slate-600 text-slate-800 flex items-center gap-2">
                                        <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                                        <div className="w-2 h-2 bg-slate-500 dark:bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                                    </div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>
                    </div>
                   <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700 rounded-b-xl">
                       <div className="flex items-center gap-2">
                           <input 
                                type="text" 
                                value={userInput} 
                                onChange={(e) => setUserInput(e.target.value)} 
                                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                                placeholder="Ask a question..." 
                                className="w-full p-2 border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-md focus:ring-2 focus:ring-sky-500 focus:outline-none"
                           />
                           <button onClick={handleSendMessage} disabled={isLoading} className="p-2 bg-sky-600 text-white rounded-full hover:bg-sky-700 disabled:bg-slate-400">
                               <Send className="w-5 h-5"/>
                           </button>
                       </div>
                   </div>
                </div>
            </div>
             <button onClick={() => setIsOpen(!isOpen)} className="mt-4 float-right bg-sky-600 text-white p-4 rounded-full shadow-lg hover:bg-sky-700 transition-transform hover:scale-110">
                <MessageCircle className="w-8 h-8"/>
            </button>
        </div>
    );
};
