from fastdtw import fastdtw

def construct_clip_reward(self, ep_frames, window_size: int = 32) -> List:
    """
    Compute dtw distance between current episode video and most similar demo
    based on clip features
    """
    rew = []
    ep_feats = [self.extract_feat(Image.fromarray(frame)) for frame in ep_frames]
    
    window_size = min(window_size, len(ep_frames))
    for start_idx in range(len(ep_frames)-window_size+1):
        # find the most similar demo
        frame_sim = [cosine_similarity(ep_feats[start_idx], feats[0]) for feats in self.demo_feats]
        idx = np.argmax(frame_sim)
        
        # calculate distance
        dtw_dist, _ = fastdtw(ep_feats[start_idx:start_idx+window_size], self.demo_feats[idx], dist=euclidean)
        rew.append(1.0 / (1.0 + dtw_dist) - 10.0)

    return rew