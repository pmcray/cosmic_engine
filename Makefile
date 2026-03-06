.PHONY: help install test clean examples examples-biosim examples-factory examples-chronos all test-biosim test-factory test-chronos

help:
	@echo "L-System Core + Bio-Sim + Factory + Chronos - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install           - Install dependencies"
	@echo "  test              - Run basic L-system examples"
	@echo "  test-biosim       - Run bio-simulation examples"
	@echo "  test-factory      - Test map generation (Factory Phase 3)"
	@echo "  test-chronos      - Test animation generation (Chronos Phase 4)"
	@echo "  examples          - Generate basic example outputs"
	@echo "  examples-biosim   - Generate bio-sim example outputs"
	@echo "  examples-factory  - Generate aesthetic synthesis examples"
	@echo "  examples-chronos  - Generate animation examples"
	@echo "  clean             - Remove generated files"
	@echo "  all               - Install and run all examples"

install:
	pip install -r requirements.txt

test:
	@echo "Testing simple tree..."
	python lsystem.py --file examples/simple_tree.l --iter 4 --verbose --stats
	@echo ""
	@echo "Testing parametric tree..."
	python lsystem.py --file examples/parametric_tree.l --iter 5 --verbose --stats
	@echo ""
	@echo "Testing algae..."
	python lsystem.py --file examples/algae.l --iter 7 --verbose --print-string

examples:
	@echo "Generating example outputs..."
	python lsystem.py --file examples/simple_tree.l --iter 4 --out output/simple_tree.obj --stats
	python lsystem.py --file examples/parametric_tree.l --iter 5 --out output/parametric_tree.obj --stats
	python lsystem.py --file examples/stochastic_plant.l --iter 4 --out output/stochastic_plant.obj --stats
	python lsystem.py --file examples/3d_tree.l --iter 3 --out output/3d_tree.obj --stats
	python lsystem.py --file examples/conditional.l --iter 4 --out output/conditional.obj --stats
	@echo "Done! Output files in output/ directory"

clean:
	rm -rf output/*.obj output/*.ply
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test-biosim:
	@echo "Testing bio-simulation examples..."
	@echo ""
	@echo "1. Tropism tree (gravitropism)..."
	python biosim.py --file examples/tropism_tree.l --iter 4 --verbose --stats
	@echo ""
	@echo "2. Phototropic plant..."
	python biosim.py --file examples/phototropic_plant.l --iter 3 --shadows --verbose

examples-biosim:
	@echo "Generating bio-sim example outputs..."
	@mkdir -p output/biosim
	python biosim.py --file examples/tropism_tree.l --iter 4 --out output/biosim/tropism_tree.obj --stats
	python biosim.py --file examples/phototropic_plant.l --iter 3 --shadows --out output/biosim/phototropic_plant.obj --stats
	python biosim.py --file examples/metabolic_tree.l --iter 5 --shadows --metabolism --out output/biosim/metabolic_tree.obj --stats
	python biosim.py --file examples/prehistoric_mega_flora.l --iter 4 --shadows --out output/biosim/prehistoric.obj --cylinders --stats
	python biosim.py --file examples/alien_structure.l --iter 4 --shadows --out output/biosim/alien.obj --stats
	@echo "Done! Bio-sim outputs in output/biosim/ directory"

test-factory:
	@echo "Testing Factory (Aesthetic Synthesis)..."
	@echo ""
	@echo "Checking system capabilities..."
	python factory.py --system-info
	@echo ""
	@echo "Listing available styles..."
	python factory.py --list-styles
	@echo ""
	@echo "Generating depth maps only (no AI required)..."
	python factory.py --input output/simple_tree.obj --maps-only --output factory_output/test --angles iso

examples-factory:
	@echo "Note: Factory examples require Stable Diffusion models (large downloads)"
	@echo "Generating map examples (no AI)..."
	@mkdir -p factory_output/maps
	python factory.py --input output/3d_tree.obj --maps-only --output factory_output/maps --angles front,iso,top --save-maps
	@echo "Maps generated in factory_output/maps/"
	@echo ""
	@echo "To generate AI images, ensure you have:"
	@echo "  - CUDA-capable GPU (recommended)"
	@echo "  - diffusers, transformers, accelerate installed"
	@echo "  - Sufficient disk space (~20GB for SDXL models)"
	@echo ""
	@echo "Example AI generation command:"
	@echo "  python factory.py --input output/3d_tree.obj --style prehistoric --output gallery/"

test-chronos:
	@echo "Testing Chronos (Animation Engine)..."
	@echo ""
	@echo "Checking video encoding capabilities..."
	python chronos.py --system-info
	@echo ""
	@echo "Generating keyframes only (fast test)..."
	python chronos.py --file examples/timed/growing_branch.l --iter 3 --keyframes --output chronos_output/test_keyframes --verbose
	@echo ""
	@echo "Generating depth map sequence..."
	python chronos.py --file examples/timed/simple_tree_anim.l --iter 3 --duration 6 --fps 10 --maps-only --output chronos_output/test_maps --verbose

examples-chronos:
	@echo "Generating Chronos animation examples..."
	@echo ""
	@echo "Note: Full video generation requires GPU and AI models"
	@echo "Generating depth map sequences and keyframes..."
	@mkdir -p chronos_output/animations
	@echo ""
	@echo "1. Growing branch (2D)..."
	python chronos.py --file examples/timed/growing_branch.l --iter 4 --duration 8 --fps 15 --maps-only --output chronos_output/animations/growing_branch --camera iso
	@echo ""
	@echo "2. Simple tree animation (3D)..."
	python chronos.py --file examples/timed/simple_tree_anim.l --iter 4 --duration 10 --fps 15 --maps-only --output chronos_output/animations/simple_tree --camera iso
	@echo ""
	@echo "3. Spiral growth..."
	python chronos.py --file examples/timed/spiral_growth.l --iter 5 --duration 12 --fps 15 --maps-only --output chronos_output/animations/spiral --camera iso
	@echo ""
	@echo "4. Exporting keyframes..."
	python chronos.py --file examples/timed/parametric_tree_anim.l --iter 5 --keyframes --output chronos_output/animations/parametric_keyframes --verbose
	@echo ""
	@echo "Done! Animation outputs in chronos_output/animations/"
	@echo ""
	@echo "To generate AI videos, ensure you have GPU and run:"
	@echo "  python chronos.py --file examples/timed/simple_tree_anim.l --iter 4 --duration 10 --video output.mp4 --style prehistoric"

all: install examples examples-biosim
