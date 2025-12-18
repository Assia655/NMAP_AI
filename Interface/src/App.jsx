import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Zap, Shield, Lock, Send, Copy, AlertCircle, User, Bot, Check, X } from 'lucide-react';

export default function App() {
  // États pour les messages, l'entrée utilisateur, le chargement et les erreurs
  const [input, setInput] = useState('');
  //tableau d'objets message affichés dans le chat. Initialisé avec un message assistant en français de bienvenue.
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: 'Bonjour! Je suis NMAP-AI, votre assistant de génération de commandes Nmap. Comment puis-je vous aider aujourd\'hui?',
      timestamp: new Date().toLocaleTimeString()
    }
  ]);
  //état booléen indiquant si une requête est en cours de traitement.
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

const API_URL = 'http://localhost:8000';


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

const handleSubmit = async () => {
  if (!input.trim() || isLoading) return;

  const userText = input;

  const userMessage = {
    id: Date.now(),
    type: 'user',
    content: userText,
    timestamp: new Date().toLocaleTimeString()
  };

  setMessages(prev => [...prev, userMessage]);
  setInput('');
  setIsLoading(true);
  setError(null);

  try {
    const response = await fetch(`${API_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: userText
      })
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const data = await response.json();
    console.log('📥 Backend response:', data);

    if (data?.type === 'out_of_context') {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.message || 'This assistant only handles Nmap-related requests.',
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
      return;
    }

    const validationStatus = data.result?.validation?.valid;
    const assistantMessage = {
      id: Date.now() + 1,
      type: 'assistant',
      content: data.result?.explanation || data.result?.error || data.error || 'Commande générée avec succès.',
      complexity: data.complexity?.level || null,
      command: data.result?.command || '',
      explanation: data.result?.explanation || '',
      validation: validationStatus === true ? 'valid' : validationStatus === false ? 'invalid' : 'unknown',
      confidence: data.complexity?.confidence || 0,
      flags: data.result?.flags || [],
      reasoning: data.comprehension?.scores?.decision_reasoning || data.comprehension?.decision || '',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, assistantMessage]);

  } catch (err) {
    console.error('❌ Error:', err);
    setError(err.message);

    setMessages(prev => [
      ...prev,
      {
        id: Date.now() + 2,
        type: 'assistant',
        content: `❌ Erreur backend: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  } finally {
    setIsLoading(false);
  }
};


  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const copyCommand = (cmd) => {
    navigator.clipboard.writeText(cmd);
  };

  const complexityColor = (level) => {
    switch(level) {
      case 'easy': return 'bg-green-500/10 text-green-400 border-green-500';
      case 'medium': return 'bg-cyan-500/10 text-cyan-400 border-cyan-500';
      case 'hard': return 'bg-red-500/10 text-red-400 border-red-500';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500';
    }
  };

  const getValidationIcon = (status) => {
    switch(status) {
      case 'valid':
        return <Check className="w-4 h-4 text-green-400" />;
      case 'invalid':
        return <X className="w-4 h-4 text-red-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-green-400 font-mono flex flex-col">
      {/* Header */}
      <div className="border-b border-green-500/30 bg-black/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8 text-green-500 animate-pulse" />
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
                  NMAP-AI
                </h1>
                <p className="text-xs text-cyan-300">Multi-Agent Pipeline • Final Decision System</p>
              </div>
            </div>

            <div className="flex items-center space-x-4 text-xs">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full animate-pulse ${error ? 'bg-red-500' : 'bg-green-500'}`}></div>
                <span className={error ? 'text-red-400' : 'text-green-400'}>
                  {error ? 'Déconnecté' : 'Connecté'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-gray-400">5001 • 5003</span>
              </div>
              <div className="flex items-center space-x-2">
                <Lock className="w-4 h-4 text-cyan-400" />
                <span className="text-gray-400">Secure</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              style={{animation: 'fadeIn 0.3s ease-out'}}
            >
              <div className={`flex ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'} space-x-3 max-w-3xl`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
                  message.type === 'user' 
                    ? 'bg-cyan-600/20 border border-cyan-500' 
                    : 'bg-green-600/20 border border-green-500'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-5 h-5 text-cyan-400" />
                  ) : (
                    <Bot className="w-5 h-5 text-green-400" />
                  )}
                </div>

                {/* Message Content */}
                <div className={`flex-1 ${message.type === 'user' ? 'mr-3' : 'ml-3'}`}>
                  <div className="flex items-center space-x-2 mb-1">
                    <span className={`text-sm font-semibold ${
                      message.type === 'user' ? 'text-cyan-400' : 'text-green-400'
                    }`}>
                      {message.type === 'user' ? 'Vous' : 'Final Decision Agent'}
                    </span>
                    <span className="text-xs text-gray-500">{message.timestamp}</span>
                  </div>

                  <div className={`rounded-lg p-4 ${
                    message.type === 'user'
                      ? 'bg-cyan-900/20 border border-cyan-700/50'
                      : 'bg-gray-900/50 border border-green-700/30'
                  }`}>
                    <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                      {message.content}
                    </p>

                    {/* Command Display */}
                    {message.command && (
                      <div className="mt-4 space-y-3">
                        {/* Reasoning Section */}
                        {message.reasoning && (
                          <div className="bg-purple-900/20 border border-purple-600/30 rounded-lg p-3">
                            <p className="text-xs text-purple-300 leading-relaxed">
                              <span className="font-bold">🧠 Raisonnement:</span> {message.reasoning}
                            </p>
                          </div>
                        )}

                        {/* Header avec métadonnées */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Terminal className="w-4 h-4 text-green-500" />
                            <span className="text-xs font-bold text-green-400">COMMANDE FINALE VALIDÉE</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            {message.confidence > 0 && (
                              <span className="text-xs text-yellow-400">
                                Confiance: {Math.round(message.confidence * 100)}%
                              </span>
                            )}
                            {message.complexity && (
                              <span className={`text-xs px-2 py-1 border rounded ${complexityColor(message.complexity)}`}>
                                {message.complexity.toUpperCase()}
                              </span>
                            )}
                            <button
                              onClick={() => copyCommand(message.command)}
                              className="text-cyan-400 hover:text-cyan-300 transition-colors p-1 hover:bg-cyan-900/20 rounded"
                              title="Copier la commande"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        {/* Commande */}
                        <div className="bg-black/50 border border-green-600/50 rounded-lg p-3">
                          <code className="text-green-400 text-sm break-all font-bold">
                            {message.command}
                          </code>
                        </div>

                        {/* Flags utilisés */}
                        {message.flags && message.flags.length > 0 && (
                          <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3">
                            <p className="text-xs text-blue-300 font-bold mb-2">📋 Flags utilisés:</p>
                            <div className="flex flex-wrap gap-2">
                              {message.flags.map((flag, idx) => (
                                <span key={idx} className="bg-blue-900/40 px-2 py-1 rounded text-xs text-blue-300 border border-blue-500/30">
                                  {flag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Explication */}
                        {message.explanation && (
                          <div className="bg-cyan-900/20 border border-cyan-600/30 rounded-lg p-3">
                            <p className="text-xs text-cyan-300 leading-relaxed">
                              <span className="font-bold">📝 Explication:</span> {message.explanation}
                            </p>
                          </div>
                        )}

                        {/* Validation Status */}
                        {message.validation && (
                          <div className="flex items-center space-x-2 text-xs">
                            {getValidationIcon(message.validation)}
                            <span className={
                              message.validation === 'valid' ? 'text-green-400' :
                              message.validation === 'invalid' ? 'text-red-400' :
                              'text-yellow-400'
                            }>
                              Validation: {message.validation.toUpperCase()}
                            </span>
                          </div>
                        )}

                        {/* Warning */}
                        <div className="flex items-start space-x-2 text-xs text-yellow-400 bg-yellow-900/10 border border-yellow-600/20 rounded p-2">
                          <AlertCircle className="w-3 h-3 flex-shrink-0 mt-0.5" />
                          <p>Assurez-vous d'avoir les autorisations appropriées avant d'exécuter cette commande.</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Loading State */}
          {isLoading && (
            <div className="flex justify-start" style={{animation: 'fadeIn 0.3s ease-out'}}>
              <div className="flex space-x-3 max-w-3xl">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center bg-green-600/20 border border-green-500">
                  <Bot className="w-5 h-5 text-green-400" />
                </div>
                <div className="ml-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-sm font-semibold text-green-400">Multi-Agent Pipeline</span>
                  </div>
                  <div className="bg-gray-900/50 border border-green-700/30 rounded-lg p-4">
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-sm text-gray-400">① Comprehension Agent (5001)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" style={{animationDelay: '0.1s'}}></div>
                        <span className="text-sm text-gray-400">② Modèles de traitement</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                        <span className="text-sm text-gray-400">③ Final Decision Agent (5003)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Footer */}
      <div className="border-t border-green-500/30 bg-black/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          {error && (
            <div className="mb-3 p-3 bg-red-900/20 border border-red-600/30 rounded text-red-400 text-xs">
              ⚠️ {error}
            </div>
          )}
          <div className="relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Décrivez votre besoin de scan en langage naturel... (Entrée pour envoyer, Shift+Entrée pour nouvelle ligne)"
              className="w-full bg-gray-900 border border-green-600 rounded-lg p-4 pr-14 text-green-300 placeholder-gray-600 focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50 resize-none"
              rows="3"
              disabled={isLoading}
            />
            
            <button
              onClick={handleSubmit}
              disabled={isLoading || !input.trim()}
              className="absolute bottom-4 right-4 bg-gradient-to-r from-green-600 to-cyan-600 hover:from-green-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-600 text-white p-3 rounded-lg transition-all duration-300 shadow-lg disabled:cursor-not-allowed"
              title="Envoyer"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>

          <div className="mt-2 text-xs text-gray-500 text-center">
            Pipeline: User → Comprehension Agent (5001) → Modèles → Final Decision Agent (5003) → Client
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
