import React, { useState, FormEvent, ChangeEvent } from 'react';
import { Compass, Loader2, BookOpen, GraduationCap, ChevronDown, ChevronUp } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import ReactMarkdown from 'react-markdown';
import { cn } from '../lib/utils';

export default function ExploreBlue() {
  const [query, setQuery] = useState<string>('');
  const [levels, setLevels] = useState<number[]>([100, 200, 300, 400]);
  const [recommendations, setRecommendations] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  const handleLevelChange = (level: number) => {
    setLevels(prev => 
      prev.includes(level) 
        ? prev.filter(l => l !== level)
        : [...prev, level]
    );
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setRecommendations('');
    try {
      // Backend API endpoint
      const response = await fetch('/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query, 
          levels: levels.length > 0 ? levels : null 
        }),
      });

      if (response.status === 429) {
        // Handle rate-limiting error
        throw new Error('Rate limit exceeded. Please try again later.');
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      let fullText = '';
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        fullText += text;
      }

      // Process the text to properly format markdown
      const formattedText = fullText
        .replace(/\\n/g, '\n\n')
        .replace(/\n\n\n+/g, '\n\n')
        .replace(/^"|"$/g, '')
        .trim();

      setRecommendations(formattedText);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const suggestions = [
    "I enjoyed my Introduction to Machine Learning class. What related courses should I take next?",
    "I want to work in sustainable energy. What courses would help me prepare for this field?",
    "I'm interested in addressing climate change through engineering. What would you recommend?",
    "I like working with data and statistics. What courses would develop these skills further?",
    "I want to learn more about robotics and automation. What courses are available?"
  ];

  // Function to apply a suggestion to the input
  const applySuggestion = (suggestion: string) => {
    setQuery(suggestion);
    setIsHelpOpen(false); // Close the accordion after selection
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-12">
          <div className="inline-flex items-center justify-center p-4 bg-blue-100 rounded-full">
            <Compass className="h-10 w-10 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            ExploreBlue
          </h1>
          <p className="text-xl text-gray-600">
            Discover steps for your academic path
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white shadow-xl rounded-2xl overflow-hidden">
          <div className="p-6 sm:p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="query" className="block text-sm font-medium text-gray-700">
                  What are you interested in learning about?
                </label>
                <div className="relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <BookOpen className="h-5 w-5 text-gray-400" aria-hidden="true" />
                  </div>
                  <input
                    id="query"
                    type="text"
                    value={query}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
                    placeholder="E.g., I am interested in machine learning and data science..."
                    className="block w-full pl-10 pr-12 py-3 border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>


                {/* Accordion Help Section */}
                <div className="mt-2">
                  <button
                    type="button"
                    onClick={() => setIsHelpOpen(!isHelpOpen)}
                    className="flex items-center text-sm text-blue-600 hover:text-blue-800 focus:outline-none transition-colors"
                  >
                    {isHelpOpen ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
                    Getting started? Please read
                  </button>
                  
                  {isHelpOpen && (
                    <div className="mt-3 bg-blue-50 p-4 rounded-lg animate-fadeIn">
                      <h4 className="font-medium text-blue-800 mb-2">What is ExploreBlue?</h4>
                      <ul className="space-y-3 text-sm text-gray-700">
                        ExploreBlue is a tool designed to help you discover courses that align with your interests and career goals. 
                        It provides personalized recommendations based on your input by constructing an ideal course based on your query and 
                        finding UM courses that are the most similar based on the course description and title.
                      </ul>
                      <h4 className="font-medium text-blue-800 mb-2">Try one of these approaches:</h4>
                      <ul className="space-y-3 text-sm text-gray-700">
                        <li className="flex items-start">
                          <span className="text-blue-600 mr-2">•</span>
                          <span>Mention a class you've taken and enjoyed</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-600 mr-2">•</span>
                          <span>Ask about industries you'd like to work in</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-600 mr-2">•</span>
                          <span>Describe societal challenges that are important to you</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-600 mr-2">•</span>
                          <span>Specify skills or technologies you want to learn</span>
                        </li>
                      </ul>

                      <h4 className="font-medium text-blue-800 mt-4 mb-2">Example prompts:</h4>
                      <div className="space-y-2">
                        {suggestions.map((suggestion, index) => (
                          <button
                            key={index}
                            type="button"
                            onClick={() => applySuggestion(suggestion)}
                            className="text-left w-full text-sm bg-white p-2 rounded border border-blue-200 hover:bg-blue-100 transition-colors"
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>

                      <h4 className="font-medium text-blue-800 mt-4 mb-2">Note: ExploreBlue does not use information about course requirements</h4>
                      
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <GraduationCap className="h-5 w-5 text-gray-500" />
                  <label className="block text-sm font-medium text-gray-700">
                    Course Levels (Optional)
                  </label>
                </div>
                <div className="flex flex-wrap gap-2">
                  {[100, 200, 300, 400, 500, 600, 700, 800, 900].map((level) => (
                    <button
                      key={level}
                      type="button"
                      onClick={() => handleLevelChange(level)}
                      className={cn(
                        "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200",
                        levels.includes(level)
                          ? "bg-blue-600 text-white shadow-md transform scale-105"
                          : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                      )}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className={cn(
                  "w-full py-4 rounded-lg font-medium transition-all duration-200",
                  "focus:outline-none focus:ring-4 focus:ring-blue-300",
                  "flex items-center justify-center space-x-2",
                  isLoading || !query.trim()
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg"
                )}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5" />
                    <span>Finding your courses...</span>
                  </>
                ) : (
                  <span>Explore Courses</span>
                )}
              </button>
            </form>
          </div>

          {/* Results Section */}
          {error && (
            <div className="px-6 sm:px-8 pb-6">
              <Alert variant="destructive">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            </div>
          )}

          {recommendations && (
            <div className="border-t border-gray-200 bg-gray-50 px-6 sm:px-8 py-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Recommended Courses
              </h2>
              <div className="prose prose-blue max-w-none">
                <ReactMarkdown>
                  {recommendations}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <footer className="mt-10 text-center text-sm text-gray-500">
          <p>
            <a href="https://forms.gle/DZ8RjF9E6ENQuEe88" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">
              Give us your feedback
            </a>
          </p>
          <p>© 2025 Hugh Van Deventer</p>
        </footer>
      </div>
    </div>
  );
}