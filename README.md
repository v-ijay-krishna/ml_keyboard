# ml_keyboard

A MIDI-based machine learning system that evaluates keyboard performance by analyzing musical features such as timing, groove, pitch accuracy, and dynamics from DAW-recorded MIDI performances.

## Project Overview

This project was developed to understand how a keyboard performance can be evaluated using structured MIDI data instead of raw audio. In music, humans can usually tell whether a performance sounds good or bad, but it is harder to explain exactly why. This system tries to break that judgment into measurable features such as note timing, rhythm consistency, pitch accuracy, and velocity dynamics.

A reference keyboard performance was recorded and used as the ideal version. Other performers' MIDI recordings were then compared with the reference performance. Features were extracted from the MIDI files and used to train a machine learning model that predicts performance scores similar to human ratings.

## What the System Does

- Reads MIDI files recorded through a DAW
- Extracts musical performance features from MIDI data
- Compares performer recordings with a reference performance
- Uses human-rated scores as ground truth
- Trains a Random Forest model to predict keyboard performance quality
- Provides a structured way to understand what makes a performance better or worse

## Key Features Analyzed

- Pitch accuracy
- Note timing
- Groove and rhythm consistency
- Note velocity and dynamics
- Inter-onset intervals
- Similarity with reference performance

## Tech Stack

- Python
- Mido
- NumPy
- Pandas
- Scikit-learn
- MIDI / DAW-based data collection

## Machine Learning Model

The project uses a Random Forest model because it works well with tabular feature-based data and also helps understand which features contribute most to the final performance score.

## Goal of the Project

The main goal is to build a system that can evaluate keyboard playing in a way that is closer to human musical judgment, while also giving a more technical explanation of performance quality.
