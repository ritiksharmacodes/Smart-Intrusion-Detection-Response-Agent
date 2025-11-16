// This turns a video frame into a YOLO input tensor.

export function preprocess(video, modelWidth = 640, modelHeight = 640) {
  const canvas = document.createElement("canvas");
  canvas.width = modelWidth;
  canvas.height = modelHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, modelWidth, modelHeight);

  const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
  const { data } = imageData;

  const input = new Float32Array(modelWidth * modelHeight * 3);

  for (let i = 0; i < modelWidth * modelHeight; i++) {
    input[i] = data[i * 4] / 255.0; // R
    input[i + modelWidth * modelHeight] = data[i * 4 + 1] / 255.0; // G
    input[i + modelWidth * modelHeight * 2] = data[i * 4 + 2] / 255.0; // B
  }

  return new ort.Tensor("float32", input, [1, 3, modelWidth, modelHeight]);
}
