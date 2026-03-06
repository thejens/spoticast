/**
 * Spoticast player — interleaves Spotify Web Playback SDK with HTML audio commentary.
 *
 * Queue format: [{type: "audio"|"spotify", url?: string, uri?: string}]
 * Playback proceeds sequentially through the queue; each item triggers the next
 * via an end-of-playback event (audio.onended for HTML audio, state change for Spotify).
 */

const STEP_MAP = {
  fetch:    'step-fetch',
  context:  'step-fetch',
  research: 'step-research',
  script:   'step-script',
  tts:      'step-tts',
};

class SpoticastPlayer {
  constructor() {
    this.queue               = [];
    this.totalItems          = 0;
    this.completedItems      = 0;
    this.deviceId            = null;
    this.spotifyPlayer       = null;
    this.audioEl             = document.getElementById('spoticast-audio');
    this.currentItem         = null;
    // Prevents double-firing of playNext on track end
    this._trackEndFired      = false;
    // True once the server has finished synthesizing all tracks
    this._generationComplete = true;
  }

  // ──────────────────────────────────────────────
  // Initialisation
  // ──────────────────────────────────────────────

  async init() {
    const { authenticated } = await this._apiFetch('/auth/token');
    if (!authenticated) {
      this._showState('landing');
      return;
    }
    this._showState('connected');
    this._loadSpotifySDK();
    this._loadLibrary();
    this._initLastFM();
  }

  // ── Last.fm widget ──────────────────────────────────────────────────────

  async _initLastFM() {
    try {
      const status = await this._apiFetch('/auth/lastfm/status');
      if (!status.available) return; // API keys not in env — keep widget hidden
      document.getElementById('lastfm-widget').style.display = '';
      this._renderLastFMStatus(status);
    } catch (e) {
      console.warn('Last.fm status check failed:', e);
    }

    document.getElementById('lastfm-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const username = document.getElementById('lastfm-username').value.trim();
      if (!username) return;
      const btn = document.getElementById('lastfm-connect-btn');
      btn.disabled = true;
      btn.textContent = 'Connecting…';
      try {
        const result = await this._apiFetch('/auth/lastfm/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username }),
        });
        this._renderLastFMStatus({ connected: true, ...result });
      } catch (err) {
        btn.textContent = 'Not found';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Connect'; }, 2000);
      }
    });

    document.getElementById('lastfm-disconnect-btn').addEventListener('click', async () => {
      await fetch('/auth/lastfm/disconnect', { method: 'POST' });
      this._renderLastFMStatus({ connected: false });
    });
  }

  _renderLastFMStatus(status) {
    const connected = document.getElementById('lastfm-connected');
    const disconnected = document.getElementById('lastfm-disconnected');
    if (status.connected) {
      const scrobbles = status.scrobbles ? ` · ${status.scrobbles.toLocaleString()} scrobbles` : '';
      document.getElementById('lastfm-pill-label').textContent = `@${status.username}${scrobbles}`;
      connected.style.display = 'inline-flex';
      disconnected.style.display = 'none';
    } else {
      connected.style.display = 'none';
      disconnected.style.display = 'flex';
    }
  }

  _loadSpotifySDK() {
    window.onSpotifyWebPlaybackSDKReady = () => this._initSpotifyPlayer();
    const script = document.createElement('script');
    script.src = 'https://sdk.scdn.co/spotify-player.js';
    document.head.appendChild(script);
  }

  async _initSpotifyPlayer() {
    // Always fetch a fresh token — it may have been refreshed server-side
    const { token } = await this._apiFetch('/auth/token');
    if (!token) return;

    this.spotifyPlayer = new window.Spotify.Player({
      name: 'Spoticast',
      getOAuthToken: async (cb) => {
        const { token: fresh } = await this._apiFetch('/auth/token');
        cb(fresh);
      },
      volume: 0.85,
    });

    this.spotifyPlayer.addListener('ready', ({ device_id }) => {
      this.deviceId = device_id;
      this._transferPlayback(device_id, token);
    });

    this.spotifyPlayer.addListener('not_ready', () => {
      this.deviceId = null;
    });

    this.spotifyPlayer.addListener('initialization_error', ({ message }) => {
      console.error('Spotify init error:', message);
      document.getElementById('sdk-warning').classList.add('visible');
    });

    this.spotifyPlayer.addListener('authentication_error', ({ message }) => {
      console.error('Spotify auth error:', message);
    });

    this.spotifyPlayer.addListener('account_error', () => {
      document.getElementById('sdk-warning').classList.add('visible');
    });

    this.spotifyPlayer.addListener('player_state_changed', (state) => {
      this._handleSpotifyStateChange(state);
    });

    this.spotifyPlayer.connect();
  }

  async _transferPlayback(deviceId, token) {
    try {
      await fetch('https://api.spotify.com/v1/me/player', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ device_ids: [deviceId], play: false }),
      });
    } catch (err) {
      console.warn('Transfer playback failed:', err);
    }
  }

  // ──────────────────────────────────────────────
  // Generation flow
  // ──────────────────────────────────────────────

  async generate(rawInput) {
    const parsed = this._parseInput(rawInput);
    if (!parsed) {
      this._showError('Paste a Spotify playlist link, URI, or a list of track URLs.');
      return;
    }

    this._showState('generating');

    let jobId;
    try {
      const res = await this._apiFetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });
      jobId = res.job_id;
    } catch (err) {
      this._showState('connected');
      this._showError('Failed to start generation: ' + err.message);
      return;
    }

    this._streamProgress(jobId);
  }

  _streamProgress(jobId) {
    const es = new EventSource(`/jobs/${jobId}/stream`);

    this._generationComplete = false;

    es.addEventListener('progress', (e) => {
      const { step, message } = JSON.parse(e.data);
      this._updateProgressStep(step, message);
    });

    // Intro is ready — start playback immediately with just the intro audio
    es.addEventListener('intro_ready', (e) => {
      const { url } = JSON.parse(e.data);
      this._markAllStepsDone();
      this._startPlayback([{ type: 'audio', url }]);
    });

    // Each track's commentary arrives as it finishes — append to live queue
    es.addEventListener('track_ready', (e) => {
      const { commentary_url, track_uri, total } = JSON.parse(e.data);
      this.queue.push({ type: 'audio', url: commentary_url });
      this.queue.push({ type: 'spotify', uri: track_uri });
      // intro (1) + N tracks × 2 items each
      this.totalItems = 1 + total * 2;
      this._updateProgress();
      this._updateSkipButton();
    });

    // Outro arrives after the last track — append to the live queue
    es.addEventListener('outro_ready', (e) => {
      const { url } = JSON.parse(e.data);
      this.queue.push({ type: 'audio', url });
      this.totalItems += 1;
      this._updateProgress();
      this._updateSkipButton();
    });

    es.addEventListener('done', (_e) => {
      es.close();
      this._generationComplete = true;
      this._updateProgress();
      this._updateSkipButton();
      // Refresh episode list so the new episode appears immediately
      this._loadEpisodes();
    });

    es.addEventListener('error', (e) => {
      es.close();
      this._generationComplete = true;
      let msg = 'Generation failed.';
      try { msg = JSON.parse(e.data).message; } catch (_) {}
      this._showState('connected');
      this._showError(msg);
    });
  }

  // ──────────────────────────────────────────────
  // Playback queue
  // ──────────────────────────────────────────────

  _startPlayback(queue) {
    this.queue          = [...queue];
    this.totalItems     = queue.length;
    this.completedItems = 0;
    this._showState('playing');
    document.getElementById('on-air-badge').classList.add('active');
    this._updateSkipButton();
    this._playNext();
  }

  _playNext() {
    if (this.queue.length === 0) {
      if (!this._generationComplete) {
        // More tracks are still being synthesized — poll until they arrive
        setTimeout(() => this._playNext(), 300);
        return;
      }
      this._onPlaybackComplete();
      return;
    }

    const item = this.queue.shift();
    this.currentItem    = item;
    this._trackEndFired = false;
    this.completedItems++;
    this._updateProgress();
    this._updateSkipButton();

    if (item.type === 'audio') {
      this._playAudio(item);
    } else if (item.type === 'spotify') {
      this._playSpotifyTrack(item);
    }
  }

  _playAudio(item) {
    this._setSegmentType('commentary');
    this._setNowPlaying('AI Commentary', 'Podcast Intro');

    // Look ahead to find what Spotify track comes next (for context)
    const nextSpotify = this.queue.find(q => q.type === 'spotify');
    if (nextSpotify?.name) {
      document.getElementById('next-up').innerHTML =
        `Up next: <strong>${nextSpotify.name}</strong>`;
    } else {
      document.getElementById('next-up').textContent = '';
    }

    document.getElementById('waveform').classList.remove('spotify-mode');
    document.getElementById('waveform').classList.remove('paused');
    document.getElementById('progress-fill').classList.remove('spotify-mode');

    this.audioEl.volume = 1;
    this.audioEl.src = item.url;
    this._crossfadeTriggered = false;

    // Fade out commentary ~2s before end so it flows smoothly into the next segment
    this.audioEl.ontimeupdate = () => {
      if (!this._crossfadeTriggered && this.audioEl.duration > 0) {
        const remaining = (this.audioEl.duration - this.audioEl.currentTime) * 1000;
        if (remaining < _CROSSFADE_MS && remaining > 0) {
          this._crossfadeTriggered = true;
          this._fadeAudioVolume(1, 0, remaining);
        }
      }
    };

    this.audioEl.onended = () => {
      this.audioEl.ontimeupdate = null;
      this._playNext();
    };

    this.audioEl.play().catch(err => {
      console.error('Audio play failed:', err);
      this._playNext();
    });
  }

  async _playSpotifyTrack(item) {
    this._setSegmentType('spotify');

    document.getElementById('waveform').classList.add('spotify-mode');
    document.getElementById('waveform').classList.remove('paused');
    document.getElementById('progress-fill').classList.add('spotify-mode');

    const { token } = await this._apiFetch('/auth/token');

    // Fetch track info — check cache first
    const cached = this._cacheGet(item.uri);
    if (cached) {
      this._setNowPlaying(cached.name, cached.artist);
      item.name = cached.name;
      item.artist = cached.artist;
      document.getElementById('next-up').textContent = '';
    } else {
      try {
        const trackId = item.uri.split(':')[2];
        const trackRes = await fetch(`https://api.spotify.com/v1/tracks/${trackId}`, {
          headers: { 'Authorization': `Bearer ${token}` },
        });
        if (trackRes.ok) {
          const trackData = await trackRes.json();
          const artist = trackData.artists.map(a => a.name).join(', ');
          this._setNowPlaying(trackData.name, artist);
          item.name = trackData.name;
          item.artist = artist;
          this._cacheSet(item.uri, { name: trackData.name, artist });
          document.getElementById('next-up').textContent = '';
        }
      } catch (_) {}
    }

    try {
      // Restore full volume whenever we start a Spotify track
      await this.spotifyPlayer.setVolume(0.85);
      await fetch(`https://api.spotify.com/v1/me/player/play?device_id=${this.deviceId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ uris: [item.uri] }),
      });
    } catch (err) {
      console.error('Spotify play failed:', err);
      this._playNext();
    }
  }

  _handleSpotifyStateChange(state) {
    if (!state) return;
    if (this.currentItem?.type !== 'spotify') return;

    const { paused, position, track_window } = state;

    // Track ended: paused, at the very start, and there's a track in history
    if (paused && position === 0 && track_window.previous_tracks.length > 0) {
      if (!this._trackEndFired) {
        this._trackEndFired = true;
        // Fade Spotify out while commentary starts (crossfade)
        this._fadeSpotifyVolume(0.85, 0, _CROSSFADE_MS).then(() => this._playNext());
      }
    }
  }

  _onPlaybackComplete() {
    document.getElementById('on-air-badge').classList.remove('active');
    document.getElementById('waveform').classList.add('paused');
    this._setNowPlaying('Episode Complete', '');
    this._setSegmentType('');
    document.getElementById('next-up').textContent = 'Thanks for listening.';
  }

  // ──────────────────────────────────────────────
  // Library loading
  // ──────────────────────────────────────────────

  _playlistOffset = 0;
  _playlistTotal = Infinity;

  async _loadLibrary() {
    this._loadEpisodes();
    this._loadRecent();
    this._loadPlaylists();
  }

  async _loadEpisodes() {
    try {
      const { episodes } = await this._apiFetch('/api/episodes');
      if (!episodes.length) return;
      const container = document.getElementById('past-episodes');
      container.innerHTML = episodes.map(ep => this._episodeCardHTML(ep)).join('');
      document.getElementById('section-episodes').classList.add('loaded');
    } catch (e) { console.warn('Failed to load past episodes:', e); }
  }

  _episodeCardHTML(ep) {
    const date = new Date(ep.created_at).toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric',
    });
    return `
      <div class="episode-card" data-episode-id="${ep.id}">
        <div class="episode-card-name">${this._esc(ep.name)}</div>
        <div class="episode-card-meta">
          ${this._esc(ep.playlist_name)} · ${ep.track_count} tracks · ${date}
        </div>
      </div>
    `;
  }

  async _playEpisode(episodeId) {
    try {
      const ep = await this._apiFetch(`/api/episodes/${episodeId}`);
      this._startPlayback(ep.queue);
    } catch (err) {
      this._showError('Could not load episode: ' + err.message);
    }
  }

  async _loadRecent() {
    try {
      const { playlists } = await this._apiFetch('/api/recent');
      if (!playlists.length) return;
      const container = document.getElementById('recent-playlists');
      container.innerHTML = playlists.map(p => this._featuredCardHTML(p)).join('');
      document.getElementById('section-recent').classList.add('loaded');
    } catch (e) { console.warn('Failed to load recent plays:', e); }
  }

  async _loadPlaylists() {
    try {
      const data = await this._apiFetch(`/api/playlists?limit=50&offset=${this._playlistOffset}`);
      this._playlistTotal = data.total;
      this._playlistOffset += data.items.length;

      const container = document.getElementById('user-playlists');
      container.insertAdjacentHTML('beforeend',
        data.items.map(p => this._playlistCardHTML(p)).join('')
      );
      document.getElementById('section-playlists').classList.add('loaded');

      const btn = document.getElementById('load-more-playlists');
      btn.style.display = this._playlistOffset < this._playlistTotal ? '' : 'none';
    } catch (e) { console.warn('Failed to load playlists:', e); }
  }

  _playlistCardHTML(p) {
    return `
      <div class="playlist-card" data-uri="${p.uri}">
        ${p.image ? `<img class="playlist-card-img" src="${p.image}" alt="" loading="lazy">` : '<div class="playlist-card-img"></div>'}
        <div class="playlist-card-body">
          <div class="playlist-card-name" title="${this._esc(p.name)}">${this._esc(p.name)}</div>
          <div class="playlist-card-meta">${p.track_count} tracks · ${this._esc(p.owner)}</div>
        </div>
      </div>
    `;
  }

  _featuredCardHTML(p) {
    return `
      <div class="featured-card playlist-card" data-uri="${p.uri}">
        ${p.image ? `<img class="featured-card-img" src="${p.image}" alt="">` : '<div class="featured-card-img"></div>'}
        <div class="featured-card-info">
          <div class="featured-card-name">${this._esc(p.name)}</div>
          <div class="featured-card-meta">${p.track_count} tracks · Updated weekly</div>
        </div>
      </div>
    `;
  }

  _esc(str) {
    const el = document.createElement('span');
    el.textContent = str ?? '';
    return el.innerHTML;
  }

  _handlePlaylistClick(uri) {
    const input = document.getElementById('playlist-uri');
    input.value = uri;
    input.focus();
    // Auto-submit
    document.getElementById('generate-form').requestSubmit();
  }

  // ──────────────────────────────────────────────
  // UI helpers
  // ──────────────────────────────────────────────

  _showState(name) {
    document.querySelectorAll('.state').forEach(el => el.classList.remove('active'));
    const el = document.getElementById(`state-${name}`);
    if (el) el.classList.add('active');
  }

  _showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.classList.add('visible');
    setTimeout(() => el.classList.remove('visible'), 6000);
  }

  _setNowPlaying(title, artist) {
    document.getElementById('now-playing-title').textContent  = title;
    document.getElementById('now-playing-artist').textContent = artist;

    const titleEl = document.getElementById('now-playing-title');
    if (title === 'AI Commentary') {
      titleEl.classList.add('italic');
    } else {
      titleEl.classList.remove('italic');
    }
  }

  _setSegmentType(type) {
    const el = document.getElementById('segment-type');
    el.className = 'segment-type ' + (type || '');
    const labels = { commentary: 'AI Commentary', spotify: 'Now Playing', '': '' };
    el.querySelector('.segment-type-label').textContent = labels[type] ?? '';
  }

  _updateProgress() {
    const pct = this.totalItems > 0
      ? Math.round((this.completedItems / this.totalItems) * 100)
      : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    const suffix = this._generationComplete ? '' : ' · generating…';
    document.getElementById('progress-label').textContent =
      `${this.completedItems} / ${this.totalItems} segments${suffix}`;
  }

  _updateSkipButton() {
    // Disable skip when there's nothing queued yet and more is still being synthesized.
    // Skipping into an empty queue while generating would stall playback.
    const blocked = this.queue.length === 0 && !this._generationComplete;
    const btn = document.getElementById('skip-btn');
    btn.disabled = blocked;
  }

  _updateProgressStep(step, message) {
    const stepId = STEP_MAP[step] ?? step;

    // Mark all previous steps as done
    const allIds = ['step-fetch', 'step-research', 'step-script', 'step-tts'];
    const idx = allIds.indexOf(stepId);
    allIds.forEach((id, i) => {
      const stepEl = document.getElementById(id);
      if (!stepEl) return;
      if (i < idx) {
        stepEl.classList.remove('active');
        stepEl.classList.add('done');
        stepEl.querySelector('.step-icon').textContent = '✓';
      } else if (i === idx) {
        stepEl.classList.add('active');
        stepEl.classList.remove('done');
        stepEl.querySelector('.step-message').textContent = message ?? '';
      }
    });
  }

  _markAllStepsDone() {
    ['step-fetch', 'step-script', 'step-tts'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.remove('active');
      el.classList.add('done');
      el.querySelector('.step-icon').textContent = '✓';
    });
  }

  // ──────────────────────────────────────────────
  // Utilities
  // ──────────────────────────────────────────────

  _parseInput(input) {
    input = (input ?? '').trim();

    // Multiple track URLs/URIs (one per line, from cmd-A cmd-C in a playlist)
    const lines = input.split(/\n/).map(l => l.trim()).filter(Boolean);
    if (lines.length > 1) {
      const trackUris = [];
      for (const line of lines) {
        const m = line.match(/spotify\.com\/track\/([a-zA-Z0-9]+)/);
        if (m) { trackUris.push(`spotify:track:${m[1]}`); continue; }
        if (/^spotify:track:[a-zA-Z0-9]+$/.test(line)) { trackUris.push(line); continue; }
        // Not a track line — bail out
        return null;
      }
      if (trackUris.length > 0) return { track_uris: trackUris };
    }

    // Single playlist URL
    const urlMatch = input.match(/spotify\.com\/playlist\/([a-zA-Z0-9]+)/);
    if (urlMatch) return { playlist_uri: `spotify:playlist:${urlMatch[1]}` };
    // Already a URI
    if (/^spotify:playlist:[a-zA-Z0-9]+$/.test(input)) return { playlist_uri: input };
    // Raw ID
    if (/^[a-zA-Z0-9]{22}$/.test(input)) return { playlist_uri: `spotify:playlist:${input}` };
    return null;
  }

  async _apiFetch(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  // ──────────────────────────────────────────────
  // Skip & crossfade
  // ──────────────────────────────────────────────

  skip() {
    if (!this.currentItem) return;
    if (this.currentItem.type === 'audio') {
      this._crossfadeTriggered = true;
      this.audioEl.ontimeupdate = null;
      this.audioEl.pause();
      this.audioEl.onended = null;
      this._playNext();
    } else if (this.currentItem.type === 'spotify') {
      if (!this._trackEndFired) {
        this._trackEndFired = true;
        this.spotifyPlayer?.pause();
        this._playNext();
      }
    }
  }

  // Ramp HTML audio element volume from → to over durationMs
  _fadeAudioVolume(from, to, durationMs) {
    const steps = Math.max(1, Math.round(durationMs / 50));
    const stepMs = durationMs / steps;
    let step = 0;
    const id = setInterval(() => {
      step++;
      this.audioEl.volume = Math.max(0, Math.min(1, from + (to - from) * (step / steps)));
      if (step >= steps) clearInterval(id);
    }, stepMs);
  }

  // Ramp Spotify player volume from → to over durationMs, returns a Promise
  async _fadeSpotifyVolume(from, to, durationMs) {
    if (!this.spotifyPlayer) return;
    const steps = 20;
    const stepMs = durationMs / steps;
    for (let i = 0; i <= steps; i++) {
      const v = Math.max(0, Math.min(1, from + (to - from) * (i / steps)));
      try { await this.spotifyPlayer.setVolume(v); } catch { /* ignore */ }
      if (i < steps) await new Promise(r => setTimeout(r, stepMs));
    }
  }

  // ──────────────────────────────────────────────
  // LocalStorage cache for Spotify metadata
  // ──────────────────────────────────────────────

  _cacheGet(uri) {
    try {
      const raw = localStorage.getItem(`sc:${uri}`);
      if (!raw) return null;
      const entry = JSON.parse(raw);
      // Expire after 7 days
      if (Date.now() - entry.ts > 7 * 86400000) {
        localStorage.removeItem(`sc:${uri}`);
        return null;
      }
      return entry.data;
    } catch { return null; }
  }

  _cacheSet(uri, data) {
    try {
      localStorage.setItem(`sc:${uri}`, JSON.stringify({ ts: Date.now(), data }));
    } catch { /* quota exceeded — ignore */ }
  }
}

// ──────────────────────────────────────────────
// Bootstrap
// ──────────────────────────────────────────────

// Duration of the volume fade at segment transitions (ms)
const _CROSSFADE_MS = 1800;

const spoticast = new SpoticastPlayer();

document.addEventListener('DOMContentLoaded', () => {
  spoticast.init();

  // Generate form submission
  document.getElementById('generate-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';
    await spoticast.generate(document.getElementById('playlist-uri').value);
    btn.disabled = false;
    btn.textContent = 'Generate Podcast';
  });

  // Auto-resize textarea and allow pasting multi-line track lists
  const textarea = document.getElementById('playlist-uri');
  const autoResize = () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
    textarea.style.overflowY = textarea.scrollHeight > 160 ? 'auto' : 'hidden';
  };
  textarea.addEventListener('input', autoResize);
  textarea.addEventListener('paste', () => setTimeout(autoResize, 0));

  // Playlist card clicks (delegated — covers recent, featured, and library)
  document.getElementById('state-connected').addEventListener('click', (e) => {
    const episodeCard = e.target.closest('.episode-card');
    if (episodeCard) {
      spoticast._playEpisode(episodeCard.dataset.episodeId);
      return;
    }
    const card = e.target.closest('.playlist-card');
    if (card) {
      spoticast._handlePlaylistClick(card.dataset.uri);
    }
  });

  // Skip button
  document.getElementById('skip-btn').addEventListener('click', () => {
    spoticast.skip();
  });

  // Load more playlists
  document.getElementById('load-more-playlists').addEventListener('click', () => {
    spoticast._loadPlaylists();
  });
});
