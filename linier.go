package ml

import (
	"log"
	"math"

	"gonum.org/v1/gonum/optimize"
)

// Linear struct of Linear regression
// this could be used for either Linear regression
// and logistic regression
type Linear struct {
	Features     [][]float64
	Theta        []float64
	Output       []float64
	LearningRate float64
	Hypothesis   LinearHypothesis
	Result       *optimize.Result
}

// LogisticRegression inherits Liner
type LogisticRegression struct {
	Linear
	TrueDegree float64
}

// LinearRegression inherits Liner
type LinearRegression struct {
	Linear
}

// LinearHypothesis struct for hypothesis
type LinearHypothesis func(X, theta []float64) float64

// LinearSetting struct for setting
type LinearSetting struct {
	MajorIteration int
	Threshod       float64
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// LinearDefaultSetting returns default
// setting for Linear regression
func LinearDefaultSetting() *LinearSetting {
	return &LinearSetting{
		MajorIteration: 1e5,
		Threshod:       1e-12,
	}
}

/***********************
 * LOGISTIC REGRESSION *
 ***********************/

// NewLogisticRegression return new pointer of
// LogisticRegression struct with default Linear
// hypothesis ax+b
func NewLogisticRegression() *LogisticRegression {
	lr := &LogisticRegression{}

	lr.Hypothesis = func(X, theta []float64) float64 {
		hypothesis := 0.0
		for key, x := range X {
			hypothesis += theta[key] * x
		}
		return hypothesis
	}
	lr.LearningRate = 1
	lr.TrueDegree = 0.5

	return lr
}

func (l *LogisticRegression) calculateCost(X []float64, y float64) float64 {
	h := sigmoid(l.Hypothesis(X, l.Theta))
	return -y*math.Log(h) - (1-y)*math.Log(1-h)
}

// Minimize start training of hypothesis
// Minimize start training of hypothesis
func (l *LogisticRegression) Minimize(setting *LinearSetting) *optimize.Result {
	var s *optimize.Settings

	if setting != nil {
		s = &optimize.Settings{
			GradientThreshold: setting.Threshod,
			MajorIterations:   setting.MajorIteration,
			Converger: &optimize.FunctionConverge{
				Absolute:   1e-12,
				Iterations: 1e5,
			},
		}
	}

	prob := optimize.Problem{
		Func: l.Func,
		Grad: l.Grad,
	}

	meth := &optimize.BFGS{}

	result, err := optimize.Minimize(prob, l.Theta, s, meth)
	if err != nil {
		log.Fatal(err)
	}

	if err = result.Status.Err(); err != nil {
		log.Fatal(err)
	}

	l.Result = result
	l.Theta = result.X

	return result
}

// Func returns cost of theta
func (l *LogisticRegression) Func(theta []float64) float64 {
	m := float64(len(l.Features))
	sum := 0.0
	for i, X := range l.Features {
		sum += l.calculateCost(X, l.Output[i])
	}

	return (1 / m) * sum
}

// Grad updates initil thetas to minimum
func (l *LogisticRegression) Grad(grad, theta []float64) {
	m := float64(len(l.Features))
	for j := range theta {
		sum := 0.0

		for i, x := range l.Features {
			sum += (sigmoid(l.Hypothesis(x, theta)) - l.Output[i]) * x[j]
		}
		grad[j] = l.LearningRate / m * sum
	}
}

// Predict start training of hypothesis
func (l *LogisticRegression) Predict(X []float64) bool {
	if l.TrueDegree == 0 {
		return sigmoid(l.Hypothesis(X, l.Theta)) >= 0.5
	}
	return sigmoid(l.Hypothesis(X, l.Theta)) >= l.TrueDegree
}

/***********************
 * Linear REGRESSION *
 ***********************/

// NewLinearRegression return new pointer of
// LogisticRegression struct with default Linear
// hypothesis ax+b
func NewLinearRegression() *LinearRegression {
	lr := &LinearRegression{}

	lr.Hypothesis = func(X, theta []float64) float64 {
		hypothesis := 0.0
		for key, x := range X {
			hypothesis += theta[key] * x
		}
		return hypothesis
	}
	lr.LearningRate = 1

	return lr
}

func (l *LinearRegression) calculateCost(x []float64, y float64) float64 {
	cost := l.Hypothesis(x, l.Theta) - y
	return math.Pow(cost, 2)
}

// Minimize start training of hypothesis
func (l *LinearRegression) Minimize(setting *LinearSetting) *optimize.Result {
	var s *optimize.Settings

	if setting != nil {
		s = &optimize.Settings{
			GradientThreshold: setting.Threshod,
			MajorIterations:   setting.MajorIteration,
			Converger: &optimize.FunctionConverge{
				Absolute:   1e-12,
				Iterations: 1e5,
			},
		}
	}

	prob := optimize.Problem{
		Func: l.Func,
		Grad: l.Grad,
	}

	meth := &optimize.BFGS{}

	result, err := optimize.Minimize(prob, l.Theta, s, meth)
	if err != nil {
		log.Fatal(err)
	}

	if err = result.Status.Err(); err != nil {
		log.Fatal(err)
	}

	l.Theta = result.X
	l.Result = result

	return result
}

// Func return cost
func (l *LinearRegression) Func(theta []float64) float64 {
	sum := 0.0
	for i, x := range l.Features {
		sum += l.calculateCost(x, l.Output[i])
	}
	m := float64(len(l.Features))

	return 1 / (2 * m) * sum
}

// Grad updates initil thetas to minimum
func (l *LinearRegression) Grad(grad, theta []float64) {
	m := float64(len(l.Features))
	for j := range theta {
		sum := 0.0

		for i, x := range l.Features {
			sum += (l.Hypothesis(x, theta) - l.Output[i]) * x[j]
		}
		grad[j] -= l.LearningRate / m * sum
	}
}

// Predict start training of hypothesis
func (l *LinearRegression) Predict(X []float64) float64 {
	return l.Hypothesis(X, l.Theta)
}
