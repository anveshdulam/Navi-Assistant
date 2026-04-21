interface AriaStatusProps {
  message: string;
}

export default function ARIAStatus({ message }: AriaStatusProps) {
  return (
    <div role="status" aria-live="polite" className="sr-only">
      {message}
    </div>
  );
}
