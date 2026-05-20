# Street-Level Building Intelligence from Google Street View
Technical Assignment Submission for Geokno India Pvt. Ltd.

Role: Senior Data Scientist

## Overview

### This project implements an end-to-end Street-Level Building Intelligence Pipeline that:

- Scrapes Google Street View imagery
- Associates each image with a building footprint
- Extracts building-level attributes using computer vision
- Visualizes results on an interactive map

### The system is designed for representative streets in:

Gachibowli
Hyderabad, India

### The final output is an interactive browser-based application where users can click a building and inspect:

- Street View image
- Estimated floor count
- Estimated façade area
- Property type classification
- Other Meta data

## Project Architecture

                ┌─────────────────────┐
                │ Input Coordinates   │
                └─────────┬───────────┘
                          │
                          ▼
            ┌────────────────────────┐
            │ Download Street View   │
            │ Imagery                │
            └─────────┬──────────────┘
                      │
                      ▼
              ┌─────────────────────┐
              │ Fetch Building OSM  │
              │ Footprints          │
              └─────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Compute Camera       │
             │ Heading & Alignment  │
             └─────────┬────────────┘
                       │
                       ▼
           ┌──────────────────────────┐
           │ CV-Based Attribute       │
           │ Extraction               │
           └─────────┬────────────────┘
                     │
                     ▼
           ┌──────────────────────────┐
           │ Interactive Map          │
           │ Visualization            │
           └──────────────────────────┘