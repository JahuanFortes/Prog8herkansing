import "./style.css";
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
// import kNear from "./knear.js";
// import jsonData from "./dataCollection/data.json";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
// const trainModelBtn = document.getElementById("trainModelBtn");
const predictBtn = document.getElementById("predictBtn");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);
const predictionDisplay = document.getElementById("prediction");
const videoUrlInput = document.getElementById("videoUrl");
const loadVideoButton = document.getElementById("loadVideo");

// Add prediction display element
const predictionDisplayElement = document.createElement("div");
predictionDisplayElement.id = "prediction";
predictionDisplayElement.style.position = "absolute";
predictionDisplayElement.style.top = "10px";
predictionDisplayElement.style.left = "10px";
predictionDisplayElement.style.background = "rgba(0,0,0,0.7)";
predictionDisplayElement.style.color = "white";
predictionDisplayElement.style.padding = "10px";
predictionDisplayElement.style.borderRadius = "5px";
predictionDisplayElement.style.fontSize = "24px";
document.querySelector(".videoView").appendChild(predictionDisplayElement);

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let image = document.querySelector("#myimage");
let player = undefined;
let lastGesture = null;
let gestureCooldown = false;
let lastPredictions = [];
const PREDICTION_HISTORY = 5; // Number of predictions to average
const MIN_CONFIDENCE = 75;
const MAX_CONFIDENCE = 95; // Cap maximum confidence

// const k = 3;
// const knn = new kNear(k);
const nn = ml5.neuralNetwork({ task: "classification", debug: true });
ml5.setBackend("webgl");

const modelDetails = {
  model: "../training-data/model/model.json",
  metadata: "../training-data/model/model_meta.json",
  weights: "../training-data/model/model.weights.bin",
};
nn.load(modelDetails, () => console.log("het model is geladen!"));

// trainModelBtn.addEventListener("click", (e) => train(e));
predictBtn.addEventListener("click", (e) => classify(e));

// Load YouTube IFrame API
const tag = document.createElement("script");
tag.src = "https://www.youtube.com/iframe_api";

const firstScriptTag = document.getElementsByTagName("script")[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

// Initialize YouTube player
window.onYouTubeIframeAPIReady = function () {
  console.log("YouTube API Ready");
  player = new YT.Player("player", {
    height: "360",
    width: "640",
    videoId: "", // Start with empty video
    playerVars: {
      playsinline: 1,
      controls: 1,
    },
    events: {
      onReady: onPlayerReady,
      onStateChange: onPlayerStateChange,
    },
  });
};

function onPlayerReady(event) {
  console.log("YouTube player ready");
  loadVideoButton.disabled = false;
  // Test player controls
  console.log("Player controls available:", {
    playVideo: typeof player.playVideo === "function",
    pauseVideo: typeof player.pauseVideo === "function",
    seekTo: typeof player.seekTo === "function",
    getCurrentTime: typeof player.getCurrentTime === "function",
  });
}

function onPlayerStateChange(event) {
  console.log("Player state changed:", event.data);
}

// Handle video loading
loadVideoButton.addEventListener("click", () => {
  const videoId = getVideoId(videoUrlInput.value);
  if (videoId) {
    console.log("Loading video:", videoId);
    player.loadVideoById(videoId);
    videoUrlInput.value = ""; // Clear input after loading
  } else {
    alert("Please enter a valid YouTube URL or Video ID");
  }
});

// Extract video ID from URL. found on StackOverflow
function getVideoId(url) {
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[2].length === 11 ? match[2] : url;
}

/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
  console.log("model loaded, you can start webcam");

  enableWebcamButton.addEventListener("click", (e) => enableCam(e));
  logButton.addEventListener("click", (e) => logAllHands(e));
};
/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
  webcamRunning = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      canvasElement.style.width = video.videoWidth;
      canvasElement.style.height = video.videoHeight;
      canvasElement.width = video.videoWidth;
      canvasElement.height = video.videoHeight;
      document.querySelector(".videoView").style.height =
        video.videoHeight + "px";
      predictWebcam();
    });
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
}

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
  results = await handLandmarker.detectForVideo(video, performance.now());

  let hand = results.landmarks[0];

  if (hand) {
    // Make prediction
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat();

    let prediction = await nn.classify(mdArr);

    // Takes the Confidence and * 100
    const label = prediction[0].label;

    let confidence = (prediction[0].confidence * 100).toFixed(1);

    // Cap maximum confidence to prevent errors
    confidence = Math.min(confidence, MAX_CONFIDENCE);

    // Adds the current prediction to an prediction history array
    lastPredictions.push({ label, confidence });
    if (lastPredictions.length > PREDICTION_HISTORY) {
      lastPredictions.shift();
    }

    // Counts how many times each label appears in the prediction history
    // Calculate average confidence and most common label
    const avgConfidence =
      lastPredictions.reduce((sum, p) => sum + parseFloat(p.confidence), 0) /
      lastPredictions.length;

    const labelCounts = {};
    lastPredictions.forEach((p) => {
      labelCounts[p.label] = (labelCounts[p.label] || 0) + 1;
    });

    const mostCommonLabel = Object.entries(labelCounts).sort(
      (a, b) => b[1] - a[1]
    )[0][0];

    // Only show prediction if we have enough history and confidence is stable
    if (
      lastPredictions.length === PREDICTION_HISTORY &&
      avgConfidence > MIN_CONFIDENCE
    ) {
      predictionDisplay.textContent = `${mostCommonLabel} (${avgConfidence.toFixed(
        1
      )}%)`;
      console.log(
        "Stable prediction:",
        mostCommonLabel,
        "with average confidence:",
        avgConfidence.toFixed(1)
      );
      handleGesture(mostCommonLabel);
    } else {
      predictionDisplay.textContent = "No clear gesture";
    }

    let thumb = hand[0];
    if (image) {
      image.style.transform = `translate(${
        video.videoWidth - thumb.x * video.videoWidth
      }px, ${thumb.y * video.videoHeight}px)`;
    }
  } else {
    predictionDisplay.textContent = "No hand detected";
    lastGesture = null;
    lastPredictions = []; // Reset prediction history when no hand is detected
  }

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  for (let hand of results.landmarks) {
    drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
      // color: "#00FF00",
      lineWidth: 5,
    });
    drawUtils.drawLandmarks(hand, {
      radius: 4,
      // color: "#FF0000",
      lineWidth: 2,
    });
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}
/********************************************************************
// LOG HAND COORDINATES IN THE CONSOLE
********************************************************************/
function logAllHands() {
  const hand = results.landmarks;
  hand.forEach((hand) => {
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat();
    console.log(mdArr);
  });
}

/********************************************************************
// MAKES A PREDICTION IN THE CONSOLE
********************************************************************/
async function classify() {
  results = await handLandmarker.detectForVideo(video, performance.now());

  let hand = results.landmarks[0];

  if (hand) {
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat();

    let prediction = await nn.classify(mdArr);
    console.log(prediction[0]);

    // Display the prediction
    const label = prediction[0].label;
    const confidence = (prediction[0].confidence * 100).toFixed(1);
    predictionDisplay.textContent = `${label} (${confidence}%)`;
  } else {
    predictionDisplay.textContent = "No hand detected";
  }
}

// Handle hand gestures
function handleGesture(gesture) {
  console.log(
    "Handling gesture:",
    gesture,
    "Player state:",
    player ? "ready" : "not ready"
  );

  if (!player || !player.playVideo) {
    console.log("YouTube player not ready or controls not available");
    return;
  }

  if (gestureCooldown) {
    console.log("Gesture on cooldown");
    return;
  }

  if (gesture !== lastGesture) {
    console.log("Processing new gesture:", gesture);
    lastGesture = gesture;
    gestureCooldown = true;

    try {
      switch (gesture) {
        case "UP":
          console.log("Playing video");
          player.playVideo();
          break;

        case "DOWN":
          console.log("Pausing video");
          player.pauseVideo();
          break;

        case "LEFT":
          const currentTime = player.getCurrentTime();
          console.log("Seeking backward from:", currentTime);
          player.seekTo(Math.max(0, currentTime - 10), true);
          break;

        case "RIGHT":
          const currentTime2 = player.getCurrentTime();
          console.log("Seeking forward from:", currentTime2);
          player.seekTo(currentTime2 + 10, true);
          break;
      }
    } catch (error) {
      console.error("Error controlling video:", error);
    }

    // Reduced cooldown to 500ms for better responsiveness
    setTimeout(() => {
      gestureCooldown = false;
      console.log("Gesture cooldown reset");
    }, 500);
  }
}

/********************************************************************
// START THE APP
********************************************************************/

if (navigator.mediaDevices?.getUserMedia) {
  createHandLandmarker();
}
