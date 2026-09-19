# api/tracing_app.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = FastAPI(title="MNIST Traced ONNX Classifier")

FastAPIInstrumentor.instrument_app(app)

ORT_SESSION = ort.InferenceSession("models/tensorflow_cnn.onnx")
INPUT_NAME = ORT_SESSION.get_inputs()[0].name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tracer.start_as_current_span("image_preprocessing"):
        image = Image.open(file.file)
        image = image.convert("L").resize((28, 28))
        image = ImageOps.invert(image)
        processed = np.array(image, dtype=np.float32) / 255.0
        processed = processed.reshape(1, 28, 28, 1)

    with tracer.start_as_current_span("onnx_inference"):
        outputs = ORT_SESSION.run(None, {INPUT_NAME: processed})
        probabilities = outputs[0][0]
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
    
    return {
        "digit": digit,
        "confidence": confidence
    }