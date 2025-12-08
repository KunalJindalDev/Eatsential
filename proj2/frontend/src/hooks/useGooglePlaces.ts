import { useQuery } from "@tanstack/react-query";

export function useGooglePlaceSearch(query?: string | null) {
  return useQuery({
    queryKey: ["google-place", query],
    queryFn: async () => {
      if (!query) return null;
      const res = await fetch(`/api/maps/search?query=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error("Google Maps lookup failed");
      const arr = await res.json();
      return arr[0] ?? null; // pick the closest/top result
    },
    enabled: !!query,
  });
}
