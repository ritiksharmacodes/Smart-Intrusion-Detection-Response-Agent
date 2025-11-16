import * as ort from "onnxruntime-web";

export class YoloDetector {
  constructor() {
    this.session = null;
  }

  async init() {
    ort.env.wasm.wasmPaths = "/ort-wasm/";
    ort.env.wasm.jsep = true; // new builds require this
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;

    this.session = await ort.InferenceSession.create("/model/yolov8n.onnx", {
      executionProviders: ["wasm"]
    });

    console.log("YOLO model loaded");
  }

  async detect(inputTensor) {
    if (!this.session) return [];

    const feeds = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    return output.data;
  }
}
