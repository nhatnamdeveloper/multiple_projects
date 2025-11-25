import { create } from 'zustand';

const useAppStore = create((set) => ({
  isLoading: false,
  message: null,
  predictionResult: null,
  
  setLoading: (status) => set({ isLoading: status }),
  setMessage: (msg) => set({ message: msg }),
  setPredictionResult: (result) => set({ predictionResult: result }),
  clearState: () => set({ isLoading: false, message: null, predictionResult: null }),
}));

export default useAppStore;
