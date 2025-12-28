class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this.chunkSize = Math.max(128, opts.chunkSize || 1024);
    this.buffer = new Float32Array(this.chunkSize);
    this.writeIndex = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;
    const output = outputs[0];
    if (output && output[0]) {
      output[0].set(channel);
    }

    let offset = 0;
    while (offset < channel.length) {
      const space = this.chunkSize - this.writeIndex;
      const remaining = channel.length - offset;
      const toCopy = Math.min(space, remaining);
      this.buffer.set(channel.subarray(offset, offset + toCopy), this.writeIndex);
      this.writeIndex += toCopy;
      offset += toCopy;
      if (this.writeIndex >= this.chunkSize) {
        const chunk = new Float32Array(this.chunkSize);
        chunk.set(this.buffer);
        this.port.postMessage(chunk, [chunk.buffer]);
        this.writeIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
