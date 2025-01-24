import React, { useState, FormEvent, ChangeEvent } from 'react';
import { Search, Loader2, School } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import ReactMarkdown from 'react-markdown';

export default function CourseRecommender() {
  const [query, setQuery] = useState<string>('');
  const [levels, setLevels] = useState<number[]>([]);
  const [recommendations, setRecommendations] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

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
      const response = await fetch('/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Send null for levels if none are selected
        body: JSON.stringify({ 
          query, 
          levels: levels.length > 0 ? levels : null 
        }),
      });

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
        .replace(/\\n/g, '\n\n')  // Replace \n with actual line breaks
        .replace(/\n\n\n+/g, '\n\n')  // Remove excessive line breaks
        .replace(/^"|"$/g, '')  // Remove leading and trailing quotation marks
        .trim();

      setRecommendations(formattedText);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <School className="mx-auto h-12 w-12 text-blue-600" />
        <h1 className="text-3xl font-bold">Course Recommender</h1>
        <p className="text-gray-600">Get personalized course recommendations based on your interests</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="query" className="block text-sm font-medium">
            What kind of courses are you looking for?
          </label>
          <div className="relative">
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
              placeholder="E.g., I'm interested in machine learning and data science..."
              className="w-full p-4 pr-12 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500"
              required
            />
            <Search className="absolute right-4 top-4 text-gray-400" />
          </div>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium">Course Levels (optional - leave unselected for all levels)</label>
          <div className="flex flex-wrap gap-2">
            {[100, 200, 300, 400, 500, 600, 700, 800, 900].map((level) => (
              <button
                key={level}
                type="button"
                onClick={() => handleLevelChange(level)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors
                  ${levels.includes(level)
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="w-full py-4 bg-blue-600 text-white rounded-lg font-medium
            hover:bg-blue-700 focus:ring-4 focus:ring-blue-500 focus:ring-opacity-50
            disabled:opacity-50 disabled:cursor-not-allowed
            flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="animate-spin" />
              <span>Getting Recommendations...</span>
            </>
          ) : (
            <span>Get Recommendations</span>
          )}
        </button>
      </form>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {recommendations && (
        <div className="bg-white p-6 rounded-lg shadow-lg space-y-4">
          <h2 className="text-xl font-semibold">Recommended Courses</h2>
          <div className="prose prose-blue max-w-none">
            <ReactMarkdown>
              {recommendations}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}