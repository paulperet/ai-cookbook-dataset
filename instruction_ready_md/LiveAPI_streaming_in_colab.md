# Streaming Audio with the Gemini Live API in Google Colab

This guide demonstrates how to use the Gemini Multimodal Live API to stream bidirectional audio within a Google Colab notebook. This setup enables real-time audio input and output, simulating a live conversation with the model.

> **Note:** This implementation requires several workarounds to function in Colab's environment. For the best live API experience, consider using [Google AI Studio](https://aistudio.google.com/app/live). To understand the core concepts of the Live API, refer to the [starter tutorial](../../quickstarts/Get_started_LiveAPI.ipynb).

## Prerequisites

### 1. Authentication
Your Google AI API key must be stored as a Colab Secret named `GOOGLE_API_KEY`.

1.  If you haven't already, create an API key in the [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  In your Colab notebook, click the key icon (🔑) in the left sidebar to open the "Secrets" pane.
3.  Add a new secret with the name `GOOGLE_API_KEY` and paste your API key as the value.

### 2. Initial Setup
Run the following cell to load your API key into the environment.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

**Important:** The first time you run the audio code, Colab will request permission to use your microphone. You must allow this permission and then **re-run the cell** for the session to start correctly.

## Step 1: Install Dependencies and Apply Backport

Colab runs Python 3.11, but the `asyncio.TaskGroup` feature we need requires a backport. The following cell installs the necessary packages and applies a monkey patch.

```python
!pip install -q websockets taskgroup

# Apply a backport for asyncio.TaskGroup in Python 3.11
import asyncio, taskgroup, exceptiongroup
asyncio.TaskGroup = taskgroup.TaskGroup
asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
```

## Step 2: Define the WebSocket Connection Handler

The following module manages the low-level WebSocket connection between the Colab Python kernel and the browser's JavaScript frontend. It handles message polling, sending, and receiving.

```python
import asyncio, contextlib, json
from google.colab import output
from IPython import display

# JavaScript helper function to start a session
_start_session_js = """
let start_session = (userFn) => {
  let debug = console.log;
  debug = ()=>{};

  let ctrl = new AbortController();
  let state = {
    recv: [],
    onRecv: () => {},
    send: [],
    onDone: new Promise((acc) => ctrl.signal.addEventListener('abort', () => acc())),
    write: (data) => {
      state.send.push(data);
    }
  };
  window._js_session_on_poll = (data) => {
    debug("on_poll", data);
    for (let msg of data) {
      if ('data' in msg) {
        state.recv.push(msg.data);
      }
      if ('error' in msg) {
        ctrl.abort(new Error('Remote: ' + msg.error));
      }
      if ('finish' in msg) {
        // TODO
        ctrl.abort(new Error('Remote: finished'));
      }
    }
    state.onRecv();
    let result = state.send;
    state.send = [];
    debug("on_poll: result", result);
    return result;
  };
  let connection = {
    signal: ctrl.signal,
    read: async () => {
      while(!ctrl.signal.aborted) {
        if (state.recv.length != 0) {
          return state.recv.shift();
        }
        await Promise.race([
          new Promise((acc) => state.onRecv = acc),
          state.onDone,
        ]);
      }
    },
    write: (data) => {
      state.write({'data': data});
    }
  };
  debug("starting userFn");
  userFn(connection).then(() => {
    debug("userFn finished");
    ctrl.abort(new Error("end of input"));
    state.write({'finished': true});
  },
  (e) => {
    debug("userFn error", e);
    console.error("Stream function failed", e);
    ctrl.abort(e);
    state.write({'error': '' + e});
  });
};
"""

class Connection:
  """Manages bidirectional communication with the JavaScript frontend."""
  def __init__(self):
    self._recv = []
    self._on_recv_ready = asyncio.Event()
    self._send = []
    self._on_done = asyncio.Future()

  async def write(self, data):
    self._send.append({'data': data})

  async def read(self):
    while not self._on_done.done() and not self._recv:
      self._on_recv_ready.clear()
      await self._on_recv_ready.wait()
    if self._on_done.done() and self._on_done.exception() is not None:
      raise self._on_done.exception()
    elif self._recv:
      return self._recv.pop(0)
    else:
      return EOFError('End of stream')

  def _poll(self):
    # Poll the JavaScript frontend for new messages
    res = output.eval_js(f'window._js_session_on_poll({json.dumps(self._send)})')
    self._send = []
    for r in res:
      if 'data' in r:
        self._recv.append(r['data'])
        self._on_recv_ready.set()
      elif 'error' in r:
        self._on_done.set_exception(Exception('Remote error: ' + r['error']))
        self._on_recv_ready.set()
      elif 'finished' in r:
        self._on_done.set_result(None)
        self._on_recv_ready.set()

  async def _pump(self, pump_interval):
    while not self._on_done.done():
      self._poll()
      await asyncio.sleep(pump_interval)

@contextlib.asynccontextmanager
async def RunningLiveJs(userCode, pump_interval=0.1):
  """
  Context manager to run JavaScript code connected to Colab.
  Yields a Connection object for message exchange.
  """
  c = Connection()
  output.eval_js(
      f"""
    let userFn = async (connection) => {{
      {userCode}
    }};
    {_start_session_js};
    start_session(userFn);
    1;
  """,
      ignore_result=True
  )
  t = asyncio.create_task(c._pump(pump_interval))

  def log_error(f):
    if f.exception() is not None:
      print('error: ', f.exception())

  t.add_done_callback(log_error)
  try:
    yield c
  finally:
    t.cancel()
    output.eval_js(
        """window._js_session_on_poll([{finish: true}]);""", ignore_result=True
    )
```

## Step 3: Define the Audio Session Handler

This module provides the core audio functionality. It uses the Web Audio API via an `AudioWorklet` to capture microphone input and play back audio output in the browser.

```python
import asyncio
import base64
from collections.abc import AsyncIterator
import contextlib
import dataclasses
import io
import json
import time
import wave
import numpy as np

@dataclasses.dataclass(frozen=True)
class AudioConfig:
  """Configuration for the audio stream."""
  sample_rate: int
  format: str = 'S16_LE'  # Only 16-bit signed little-endian is supported
  channels: int = 1       # Only mono audio is supported

  @property
  def sample_size(self) -> int:
    assert self.format == 'S16_LE'
    return 2

  @property
  def frame_size(self) -> int:
    return self.channels * self.sample_size

  @property
  def numpy_dtype(self) -> np.dtype:
    assert self.format == 'S16_LE'
    return np.dtype(np.int16).newbyteorder('<')

@dataclasses.dataclass(frozen=True)
class Audio:
  """A container for audio data and its configuration."""
  config: AudioConfig
  data: bytes

  @staticmethod
  def silence(config: AudioConfig, length_seconds: float | int) -> 'Audio':
    frame = b'\0' * config.frame_size
    num_frames = int(length_seconds * config.sample_rate)
    if num_frames < 0:
      num_frames = 0
    return Audio(config=config, data=frame * num_frames)

  def as_numpy(self):
    return np.frombuffer(self.data, dtype=self.config.numpy_dtype)

  def as_wav_bytes(self) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wav:
      wav.setnchannels(self.config.channels)
      wav.setframerate(self.config.sample_rate)
      assert self.config.format == 'S16_LE'
      wav.setsampwidth(2)  # 16bit
      wav.writeframes(self.data)
    return buf.getvalue()

  def _ipython_display_(self):
    """Hook for displaying audio as an HTML5 player in a notebook."""
    from IPython.display import display, HTML
    b64_wav = base64.b64encode(self.as_wav_bytes()).decode('utf-8')
    display(HTML(f"""
        <audio controls>
          <source src="data:audio/wav;base64,{b64_wav}" type="audio/wav">
        </audio>
    """.strip()))

  async def astream_realtime(
      self, expected_delta_sec: float = 0.1
  ) -> AsyncIterator[bytes]:
    """Yields audio data in chunks as if it was played in real-time."""
    current_pos = 0
    mono_start_ns = time.monotonic_ns()
    while current_pos < len(self.data):
      await asyncio.sleep(expected_delta_sec)
      delta_ns = time.monotonic_ns() - mono_start_ns
      expected_pos_frames = int(delta_ns * self.config.sample_rate / 1e9)
      next_pos = expected_pos_frames * self.config.frame_size
      if next_pos > current_pos:
        yield self.data[current_pos:next_pos]
        current_pos = next_pos

  def __add__(self, other: 'Audio') -> 'Audio':
    assert self.config == other.config
    return Audio(config=self.config, data=self.data + other.data)

class FailedToStartError(Exception):
  """Raised when the audio session fails to initialize."""

class AudioSession:
  """Manages the connection to the browser's audio recording/playback."""
  def __init__(self, config: AudioConfig, connection: Connection):
    self._config = config
    self._connection = connection
    self._done = False
    self._read_queue: asyncio.Queue[bytes] = asyncio.Queue()
    self._started = asyncio.Future()

  @property
  def config(self) -> AudioConfig:
    return self._config

  async def await_start(self):
    await self._started

  async def _read_loop(self):
    while True:
      data = await self._connection.read()
      if 'audio_in' in data:
        raw_data = base64.b64decode(data['audio_in'].encode('utf-8'))
        self._read_queue.put_nowait(raw_data)
      if 'started' in data:
        self._started.set_result(None)
      if 'failed_to_start' in data:
        self._started.set_exception(
            FailedToStartError(
                f'Failed to start audio: {data["failed_to_start"]}'
            )
        )

  async def enqueue(self, audio_data: bytes):
    b64_data = base64.b64encode(audio_data).decode('utf-8')
    await self._connection.write({'audio_out': b64_data})

  async def clear_queue(self):
    await self._connection.write({'flush': True})

  async def read(self) -> bytes:
    return await self._read_queue.get()

STANDARD_AUDIO_CONFIG = AudioConfig(sample_rate=16000, channels=1)

# JavaScript code for the AudioWorklet processor
_audio_processor_worklet_js = """
class PortProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = [];
    this.port.onmessage = (event) => {
      if ('enqueue' in event.data) {
        this.enqueueAudio(event.data.enqueue);
      }
      if ('clear' in event.data) {
        this.clearAudio();
      }
    };
    this._out = [];
    this._out_len = 0;
    console.log("PortProcessor ctor", this);
    this.port.postMessage({
      debug: "Hello from the processor!",
    });
  }

  encodeAudio(input) {
    const channel = input[0];
    const data = new ArrayBuffer(2 * channel.length);
    const view = new DataView(data);
    for (let i=0; i<channel.length; i++) {
      view.setInt16(2*i, channel[i] * 32767, true);
    }
    return data;
  }

  enqueueAudio(input) { // bytearray
    let view = new DataView(input);
    let floats = [];
    for (let i=0; i<input.byteLength; i+=2) {
      floats.push(view.getInt16(i, true) / 32768.0);
    }
    this._queue.push(Float32Array.from(floats));
  }

  dequeueIntoBuffer(output) { // Float32Array
    let idx = 0;
    while (idx < output.length) {
      if (this._queue.length === 0) {
        return;
      }
      let input = this._queue[0];
      if (input.length == 0) {
        this._queue.shift();
        continue;
      }
      let n = Math.min(input.length, output.length - idx);
      output.set(input.subarray(0, n), idx);
      this._queue[0] = input.subarray(n);
      idx += n;
    }
  }

  clearAudio() {
    this._queue = [];
  }

  process(inputs, outputs, parameters) {
    // forward input audio
    let data = this.encodeAudio(inputs[0]);
    this._out.push(data);
    this._out_len += data.byteLength;
    // only send in 50ms batches, ipykernel will die when it gets too frequent
    if (this._out_len > (2*sampleRate / 20)) {
      let concat = new Uint8Array(this._out_len);
      let idx = 0;
      for (let a of this._out) {
        concat.set(new Uint8Array(a), idx);
        idx += a.byteLength;
      }
      this._out = [];
      this._out_len = 0;
      this.port.postMessage({
        'audio_in': concat.buffer,
      });
    }

    // forward output
    this.dequeueIntoBuffer(outputs[0][0]);
    // copy to other channels
    for (let i=1; i<outputs[0].length; i++) {
      const src = outputs[0][0];
      const dst = outputs[0][i];
      dst.set(src.subarray(0, dst.length));
    }
    return true;
  }
}

registerProcessor('port-processor', PortProcessor);
"""

# JavaScript code to initialize the audio session in the browser
_audio_session_js = """
let audioCtx = new AudioContext({sampleRate: sample_rate});
await audioCtx.audioWorklet.addModule(URL.createObjectURL(
  new Blob([audio_worklet_js], {type: 'text/javascript'})
));
let userMedia;
try {
  userMedia = await navigator.mediaDevices.getUserMedia({
    audio: {sampleRate: sample_rate, echoCancellation: true, channelCount: 1},
  });
} catch (e) {
  connection.write({failed_to_start: e});
  throw e;
}
console.log("colab_audio: userMedia=", userMedia);
connection.write({started: true})

try {
  let source = audioCtx.createMediaStreamSource(userMedia);
  let processor = new AudioWorkletNode(audioCtx, 'port-processor');
  processor.port.onmessage = (event) => {
    if ('audio_in' in event.data) {
      // base64 encode ugly way
      let encoded = btoa(String.fromCharCode(
          ...Array.from(new Uint8Array(event.data.audio_in))));
      connection.write({audio_in: encoded});
    } else {
      console.log("from processor (unhandled)", event);
    }
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);

  processor.port.start();

  // Handle messages from Python to enqueue or clear audio
  while (true) {
    let msg = await connection.read();
    if ('audio_out' in msg) {
      processor.port.postMessage({enqueue: atob(msg.audio_out)});
    }
    if ('flush' in msg) {
      processor.port.postMessage({clear: true});
    }
  }
} finally {
  // Cleanup
  if (userMedia) {
    for (let track of userMedia.getTracks()) {
      track.stop();
    }
  }
  if (audioCtx) {
    await audioCtx.close();
  }
}
"""

@contextlib.asynccontextmanager
async def RunningLiveAudio(
    config: AudioConfig = STANDARD_AUDIO_CONFIG, pump_interval: float = 0.1
):
  """
  Context manager to start a live audio session.
  Yields an AudioSession object for recording and playback.
  """
  async with RunningLiveJs(
      _audio_session_js.replace('sample_rate', str(config.sample_rate)).replace(
          'audio_worklet_js', json.dumps(_audio_processor_worklet_js)
      ),
      pump_interval,
  ) as connection:
    session = AudioSession(config, connection)
    await session.await_start()
    asyncio.create_task(session._read_loop())
    yield session
```

## Step 4: Test the Audio System

Before connecting to the Gemini API, let's verify the