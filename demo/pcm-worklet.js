/* Raw PCM only. No encoded-media APIs or codecs are used. */
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.port.onmessage = ({ data }) => {
      if (data?.type === "flush") {
        this.flush();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  flush() {
    if (!this.used) return;
    const samples = this.buffer.slice(0, this.used);
    this.port.postMessage({ type: "samples", samples }, [samples.buffer]);
    this.used = 0;
  }

  process(inputs, outputs) {
    for (const channels of outputs) for (const channel of channels) channel.fill(0);
    const channels = inputs[0];
    if (channels?.length) {
      for (let frame = 0; frame < channels[0].length; frame++) {
        let value = 0;
        for (const channel of channels) value += channel[frame] || 0;
        this.buffer[this.used++] = value / channels.length;
        if (this.used === this.buffer.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
