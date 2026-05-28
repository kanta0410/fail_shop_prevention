'use client';
export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return <div><h2>Error!</h2><p>{error.message}</p></div>;
}
