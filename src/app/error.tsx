"use client";

import { Button } from "@/components/ui/button";

export default function Error({
  error: _error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background text-foreground px-6">
      <div
        className="w-full max-w-md space-y-4 text-center"
        role="alert"
        aria-live="assertive"
      >
        <h2 className="text-lg font-semibold">Something went wrong</h2>
        <p className="text-sm text-muted-foreground">
          We ran into an unexpected issue. Please try again.
        </p>
        <Button onClick={reset} aria-label="Retry loading the app">
          Try again
        </Button>
      </div>
    </div>
  );
}
