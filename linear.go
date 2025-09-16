package main

import (
	"fmt"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

func main() {
	// Example dataset: y = 2x + 1
	inputs := [][]float64{
		{1}, {2}, {3}, {4}, {5},
	}
	outputs := []float64{3, 5, 7, 9, 11}

	model := linear.NewLinearRegression(base.BatchGA, 1e-4, 0, nil)

	// Train the model
	err := model.Learn(inputs, outputs)
	if err != nil {
		panic(err)
	}

	// Predict
	prediction, err := model.Predict([]float64{6})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Prediction for x=6: %.2f\n", prediction) // ~13.00
}
