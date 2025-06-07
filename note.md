# Psuedo code

### Loading the data

- load the data from the model into the existing project
  > if model doesn't work then error.
- user starts the cam
- > user makes a pose with there hand. handLandmarker.detectfromvideo(video)
  > result =>
  > in result.landmarks (array) const hand = result.landmarks[0]
- > if the data of the pose = similar to the T.data in the model then show the perdiction in the console
- > if this is false then display a msg that something is wrong

---

### Using the data

- connect the a button on the keyboard with the data.points you just logged
- > error if this doens't work
- > if it does work find a way to press the connected keyboard btn
- > if that works then find a way to turn the keyboard btn inputs => steam btn inputs

---
