import "./style.css";
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js";
import jsonData from "./dataCollection/data.json";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const trainModelBtn = document.getElementById("trainModelBtn");
const predictBtn = document.getElementById("predictBtn");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let image = document.querySelector("#myimage");

const k = 3;
const knn = new kNear(k);
const nn = ml5.neuralNetwork({ task: "classification", debug: true });
ml5.setBackend("webgl");

trainModelBtn.addEventListener("click", (e) => train(e));
predictBtn.addEventListener("click", (e) => classify(e));
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
    knn.learn(
      [
        0.5641078352928162, 0.945195734500885, 8.909975122151081e-7,
        0.4583781957626343, 0.9103289842605591, -0.05631684884428978,
        0.3842364251613617, 0.8102291822433472, -0.07631140947341919,
        0.33829647302627563, 0.7276986241340637, -0.09070196002721786,
        0.2851669192314148, 0.6665575504302979, -0.10530006140470505,
        0.4366101622581482, 0.6058652400970459, -0.028175722807645798,
        0.39951539039611816, 0.4827485978603363, -0.04970965534448624,
        0.38076573610305786, 0.3989035487174988, -0.07153762131929398,
        0.3718416690826416, 0.3240138292312622, -0.09042521566152573,
        0.5081828236579895, 0.5792769193649292, -0.026792017742991447,
        0.5075228214263916, 0.42093902826309204, -0.04279112070798874,
        0.5135834813117981, 0.3168356120586395, -0.06562227755784988,
        0.5222140550613403, 0.22724312543869019, -0.0848212018609047,
        0.5717760324478149, 0.5930261015892029, -0.03492129221558571,
        0.5905476212501526, 0.4484345316886902, -0.06617575883865356,
        0.607276201248169, 0.35289353132247925, -0.10059545934200287,
        0.6202023029327393, 0.2590482234954834, -0.126539945602417,
        0.630470335483551, 0.631130576133728, -0.04787624254822731,
        0.6762036681175232, 0.5413030385971069, -0.08821128308773041,
        0.7102418541908264, 0.48563289642333984, -0.11540111899375916,
        0.7392791509628296, 0.4214339852333069, -0.134842187166214,
      ],
      "Light_Atk",
      [
        0.5390760898590088, 0.7805055975914001, 4.5287299599294784e-7,
        0.4859353005886078, 0.752626895904541, -0.023384204134345055,
        0.4540722072124481, 0.6766700744628906, -0.033175501972436905,
        0.48096349835395813, 0.6061881184577942, -0.04498373344540596,
        0.5140483379364014, 0.5624136328697205, -0.05564291775226593,
        0.4375342130661011, 0.547446608543396, -0.008043286390602589,
        0.4000694751739502, 0.47205230593681335, -0.022760789841413498,
        0.38076427578926086, 0.4255291521549225, -0.03542468696832657,
        0.36547937989234924, 0.3843362331390381, -0.04657283052802086,
        0.4825795292854309, 0.522761344909668, -0.014191878028213978,
        0.4748874604701996, 0.4154396057128906, -0.03036401979625225,
        0.47289174795150757, 0.35457414388656616, -0.04640169441699982,
        0.4720004200935364, 0.3048267662525177, -0.05828326940536499,
        0.5245456695556641, 0.5349391102790833, -0.02452102303504944,
        0.5232202410697937, 0.44377201795578003, -0.05520286411046982,
        0.5259904861450195, 0.48608464002609253, -0.06864908337593079,
        0.528828501701355, 0.5272775292396545, -0.07168059796094894,
        0.5666739344596863, 0.5704151391983032, -0.03662262484431267,
        0.5732385516166687, 0.5111520290374756, -0.0651286244392395,
        0.5625514388084412, 0.5441639423370361, -0.07275629043579102,
        0.5530936121940613, 0.5867138504981995, -0.0742720514535904,
      ],
      "Medium_Atk"
    );

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
function logAllHands() {
  const hand = results.landmarks;
  hand.forEach((hand) => {
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat();
    console.log(mdArr);
  });

  // console.log(mdArr);
}

// for (let hand of results.landmarks) {
//   console.log(hand);
//   console.log(results.landmarks[0]);
// }

// train
function train() {
  if (Array.isArray(jsonData)) {
    for (let pose of jsonData) {
      knn.learn(pose.points, pose.label);
      console.log(pose);
    }
  }
}

async function classify() {
  results = await handLandmarker.detectForVideo(video, performance.now());
  let hand = results.landmarks[0];
  if (hand) {
    const mdArr = hand
      .map((landmark) => [landmark.x, landmark.y, landmark.z])
      .flat();
    let perdiction = knn.classify(mdArr);
    console.log(perdiction);
  }
}
/********************************************************************
// START THE APP
********************************************************************/

if (navigator.mediaDevices?.getUserMedia) {
  createHandLandmarker();
}
