package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	DFS = iota
	BFS
	GBFS
	ASTAR
	DIJKSTRA
)

type Point struct {
	Row    int
	Column int
}

type Wall struct {
	State Point
	wall  bool
}

type Maze struct {
	Height int
	Width  int
	Start  Point
	Goal   Point
	Walls  [][]Wall
}

func main() {
	var m Maze
	var maze, searchType string

	flag.StringVar(&maze, "file", "maze.txt", "maze file")
	flag.StringVar(&searchType, "search", "dfs", "search type")
	flag.Parse()

	err := m.Load(maze)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Maze height / width : ", m.Height, m.Width)
}

func (m *Maze) Load(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		fmt.Printf("error opening %s : %s\n", fileName, err)
	}
	defer f.Close()

	var fileContents []string

	reader := bufio.NewReader(f)

	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		} else if err != nil {
			return errors.New("cannot open file")
		}
		fileContents = append(fileContents, line)
	}

	foundStart, foundEnd := false, false

	for _, line := range fileContents {
		if strings.Contains(line, "A") {
			foundStart = true
		}

		if strings.Contains(line, "B") {
			foundEnd = true
		}
	}

	if !foundStart {
		return errors.New("starting location not found")
	}

	if !foundEnd {
		return errors.New("ending location not found")
	}

	m.Height = len(fileContents)
	m.Width = len(fileContents[0])

	var rows [][]Wall

	for i, row := range fileContents {
		var cols []Wall

		for j, col := range row {
			currentLetter := fmt.Sprintf("%c", col)
			var wall Wall

			switch currentLetter {
			case "A":
				m.Start = Point{Row: i, Column: j}
				wall.State.Row = i
				wall.State.Column = j
				wall.wall = false

			case "B":
				m.Goal = Point{Row: i, Column: j}
				wall.State.Row = i
				wall.State.Column = j
				wall.wall = false

			case "":
				wall.State.Row = i
				wall.State.Column = j
				wall.wall = false

			case "#":
				wall.State.Row = i
				wall.State.Column = j
				wall.wall = true

			default:
				continue
			}

			cols = append(cols, wall)
		}
		rows = append(rows, cols)
	}

	m.Walls = rows
	return nil

}
