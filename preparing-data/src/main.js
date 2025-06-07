//#region imports
import "./style.css";
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);
const saveHandDataBtn = document.getElementById("saveHandData");
const labelCounters = {};

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let image = document.querySelector("#myimage");
//#endregion imports

addButton.addEventListener("click", addToLocalStorage);
downloadButton.addEventListener("click", downloadJSON);

/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/

//#region PoseDetector
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
//#endregion PoseDetector

/********************************************************************
// START THE WEBCAM
********************************************************************/

//#region Start Webcam
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
//#endregion Start Webcam

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
  results = await handLandmarker.detectForVideo(video, performance.now());

  let hand = results.landmarks[0];

  if (hand) {
    let thumb = hand[4];
    image.style.transform = `translate(${
      video.videoWidth - thumb.x * video.videoWidth
    }px, ${thumb.y * video.videoHeight}px)`;
    // train();
  }

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  for (let hand of results.landmarks) {
    drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5,
    });
    drawUtils.drawLandmarks(hand, {
      radius: 4,
      color: "#FF0000",
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
// Logs all detected hand landmarks to the console
function logAllHands() {
  const hand = results.landmarks; // Get array of detected hands with landmark data
  hand.forEach((hand) => {
    // For each hand, map its landmarks (x, y, z) into a flat array
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat(); // Flatten the array of coordinates into one single array
    console.log(mdArr); // Log the flattened coordinates
  });
}

// Loads label counters from localStorage and updates the UI
function loadCounters() {
  // Retrieve saved dataset from localStorage, or default to empty array
  const storedData = JSON.parse(localStorage.getItem("handDataset") || "[]");

  // Count how many times each label appears in the stored data
  storedData.forEach((entry) => {
    labelCounters[entry.label] = (labelCounters[entry.label] || 0) + 1;
  });

  // Update label selector options with counts
  updateSelectOptions();
}

// Updates the label dropdown with the current count of each label
function updateSelectOptions() {
  const options = labelSelector.options; // Get all label options from the dropdown
  for (let i = 0; i < options.length; i++) {
    const label = options[i].value; // Get the label value
    const count = labelCounters[label] || 0; // Get count or default to 0
    options[i].textContent = `${label} (${count})`; // Set option text with label and count
  }
}

// Adds a new hand entry to localStorage under the selected label
function addToLocalStorage() {
  const hands = results.landmarks; // Get current detected hands
  if (!hands || hands.length === 0) {
    alert("No hand landmarks detected.");
    return; // Exit if no hand data is available
  }

  // Convert the first detected hand's landmarks into a flat array
  const mdArr = hands[0]
    .map((landmark) => [landmark.x, landmark.y, landmark.z])
    .flat();

  const label = labelSelector.value; // Get the selected label from dropdown

  // Get existing data from localStorage, or use an empty array
  const storedData = JSON.parse(localStorage.getItem("handDataset") || "[]");

  // Create a new entry with points and label
  const entry = {
    points: mdArr,
    label: label,
  };

  storedData.push(entry); // Add the new entry to the dataset
  localStorage.setItem("handDataset", JSON.stringify(storedData)); // Save updated data

  // Update the label counter
  labelCounters[label] = (labelCounters[label] || 0) + 1;

  // Refresh the label options UI with updated count
  updateSelectOptions();
}

// Allows the user to download the saved hand dataset as a JSON file
function downloadJSON() {
  const storedData = localStorage.getItem("handDataset"); // Get the data from localStorage
  if (!storedData) {
    alert("No data to download.");
    return; // Exit if there's no data
  }

  // Create a Blob object from the JSON data
  const blob = new Blob([storedData], { type: "application/json" });

  // Create a temporary URL and an anchor element to trigger download
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "data.json"; // Set the downloaded file name
  a.click(); // Trigger the download

  URL.revokeObjectURL(url); // Clean up the temporary URL
}

// Initialize counters and dropdown when the page loads
loadCounters();

/********************************************************************
// START THE APP
********************************************************************/

if (navigator.mediaDevices?.getUserMedia) {
  createHandLandmarker();
}
