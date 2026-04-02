"""
Chess Playing Robot — Single File
==================================
Vision  → VisionSystem
Robot   → Manipulator
Engine  → ChessEngine  (Stockfish via python-chess)
UI      → ChessBoardUI (Pygame)
"""

import time
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import chess
import chess.engine
import pygame


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

ENGINE_PATH = "/usr/games/stockfish"


# ══════════════════════════════════════════════════════════════
#  VISION SYSTEM
# ══════════════════════════════════════════════════════════════

class VisionSystem:

    def __init__(self, cam_index=0, warp=800, thresh_delta=15):

        self.cap = cv2.VideoCapture(cam_index)
        self.WARP = warp
        self.THRESH_DELTA = thresh_delta

        self.points = []
        self.H = None
        self.sqdict = {}

        cv2.namedWindow("camera")
        cv2.setMouseCallback("camera", self.mouse)

    # -----------------------
    # MOUSE INPUT
    # -----------------------
    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")

    # -----------------------
    # BUILD GRID
    # -----------------------
    def build_sqdict(self):

        files = "abcdefgh"
        ranks = "87654321"
        cell = self.WARP // 8

        for r in range(8):
            for c in range(8):

                x = c * cell
                y = r * cell

                poly = [
                    (x,        y),
                    (x + cell, y),
                    (x + cell, y + cell),
                    (x,        y + cell),
                ]

                name = files[c] + ranks[r]
                self.sqdict[name] = poly

    # -----------------------
    # CALIBRATION
    # -----------------------
    def calibrate(self):

        print("\nClick the 4 corners of the board (top-left → top-right → bottom-right → bottom-left)")

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            vis = frame.copy()

            for p in self.points:
                cv2.circle(vis, p, 5, (0, 0, 255), -1)

            cv2.imshow("camera", vis)

            if len(self.points) == 4:

                src = np.float32(self.points)
                dst = np.float32([
                    [0,         0],
                    [self.WARP, 0],
                    [self.WARP, self.WARP],
                    [0,         self.WARP],
                ])

                self.H = cv2.getPerspectiveTransform(src, dst)
                self.build_sqdict()

                print("✅ Board calibrated and locked.")
                break

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow("camera")

    # -----------------------
    # GET BOARD FRAME
    # -----------------------
    def get_board_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        board = cv2.warpPerspective(frame, self.H, (self.WARP, self.WARP))

        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        return board, gray

    # -----------------------
    # SQUARE MEAN HELPER
    # -----------------------
    def square_mean(self, img, poly):
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
        return cv2.mean(img, mask=mask)[0]

    # -----------------------
    # MOVE DETECTION
    # -----------------------
    def detect_move(self, before_frame, after_frame):

        sources      = []
        destinations = []

        for sq, poly in self.sqdict.items():

            before = self.square_mean(before_frame, poly)
            after  = self.square_mean(after_frame,  poly)

            delta = after - before

            if abs(delta) < self.THRESH_DELTA:
                continue

            if delta > 0:
                sources.append((sq, delta))
            else:
                destinations.append((sq, delta))

        print("\nSources:",      sources)
        print("Destinations:", destinations)

        # Normal move / capture
        if len(sources) == 1 and len(destinations) == 1:
            return sources[0][0], destinations[0][0]

        # Castling (2 pieces lifted + 2 pieces placed)
        if len(sources) == 2 and len(destinations) == 2:

            pairs = []
            for s, _ in sources:
                for d, _ in destinations:
                    dist = abs(ord(s[0]) - ord(d[0]))
                    pairs.append((dist, s, d))

            pairs.sort(reverse=True)
            king_move = pairs[0]
            print("Castling detected")
            return king_move[1], king_move[2]

        print("⚠️  Detection failed")
        return None, None

    # -----------------------
    # SNAPSHOT
    # -----------------------
    def capture_frame(self):
        _, gray = self.get_board_frame()
        return gray.copy() if gray is not None else None

    # -----------------------
    # VERIFY ROBOT MOVE
    # -----------------------
    def verify_move(self, before, after, expected_from, expected_to):

        from_sq, to_sq = self.detect_move(before, after)

        if from_sq is None:
            print("⚠️  No valid move detected during verification")
            return False

        if from_sq == expected_from and to_sq == expected_to:
            print("✅ Move verified")
            return True

        print("⚠️  Mismatch detected")
        print(f"   Expected : {expected_from} → {expected_to}")
        print(f"   Got      : {from_sq} → {to_sq}")
        return False

    # -----------------------
    # CLEANUP
    # -----------------------
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  CHESS ENGINE  (Stockfish wrapper)
# ══════════════════════════════════════════════════════════════

class ChessEngine:

    def __init__(self, engine_path, think_time=0.5):

        self.board      = chess.Board()
        self.engine     = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.think_time = think_time

    # -----------------------
    # APPLY HUMAN MOVE (UCI)
    # -----------------------
    def apply_move_uci(self, from_sq, to_sq):

        move = chess.Move.from_uci(from_sq + to_sq)

        if move in self.board.legal_moves:
            self.board.push(move)
            print("Move applied:", move)
            return True

        print(f"⚠️  Illegal move: {from_sq} → {to_sq}")
        return False

    # -----------------------
    # GET ENGINE MOVE
    # -----------------------
    def get_best_move(self):

        result = self.engine.play(
            self.board,
            chess.engine.Limit(time=self.think_time)
        )

        move = result.move
        print("Engine move:", move)
        return move

    # -----------------------
    # APPLY ENGINE MOVE
    # -----------------------
    def apply_engine_move(self, move):
        self.board.push(move)

    # -----------------------
    # MOVE → SQUARES
    # -----------------------
    def move_to_squares(self, move):
        uci = move.uci()
        return uci[:2], uci[2:]

    # -----------------------
    # CAPTURE CHECK
    # -----------------------
    def is_capture(self, move):
        return self.board.is_capture(move)

    # -----------------------
    # GAME STATUS
    # -----------------------
    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        return self.board.result()

    # -----------------------
    # DISPLAY
    # -----------------------
    def print_board(self):
        print(self.board)

    def get_fen(self):
        return self.board.fen()

    def print_pretty_board(self):

        board = self.board
        print("\n  a b c d e f g h")

        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                square = rank * 8 + file
                piece  = board.piece_at(square)
                row.append(piece.symbol() if piece else ".")
            print(f"{rank + 1} " + " ".join(row))

        print()

    # -----------------------
    # CLEANUP
    # -----------------------
    def close(self):
        self.engine.quit()


# ══════════════════════════════════════════════════════════════
#  CHESS BOARD UI  (Pygame)
# ══════════════════════════════════════════════════════════════

class ChessBoardUI:

    def __init__(self, engine, size=600):

        self.engine = engine
        self.size   = size
        self.cell   = size // 8

        pygame.init()
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Chess Board")

        self.light = (240, 217, 181)
        self.dark  = (181, 136,  99)
        self.font  = pygame.font.SysFont("Arial", self.cell // 2)

    # -----------------------
    # DRAW BOARD
    # -----------------------
    def draw_board(self):

        for r in range(8):
            for c in range(8):
                color = self.light if (r + c) % 2 == 0 else self.dark
                pygame.draw.rect(
                    self.screen, color,
                    (c * self.cell, r * self.cell, self.cell, self.cell)
                )

    # -----------------------
    # DRAW PIECES
    # -----------------------
    def draw_pieces(self):

        board = self.engine.board

        for square in board.piece_map():

            piece  = board.piece_at(square)
            symbol = piece.symbol()

            col = square % 8
            row = 7 - (square // 8)

            text = self.font.render(symbol, True, (0, 0, 0))
            self.screen.blit(
                text,
                (col * self.cell + self.cell // 3,
                 row * self.cell + self.cell // 4)
            )

    # -----------------------
    # UPDATE DISPLAY
    # -----------------------
    def update(self):
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()

    # -----------------------
    # HANDLE EVENTS
    # -----------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True


# ══════════════════════════════════════════════════════════════
#  MANIPULATOR  (Arduino serial robot arm)
# ══════════════════════════════════════════════════════════════

class Manipulator:

    def __init__(self, port="/dev/ttyUSB0", baud=9600):

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)

        # Clear Arduino startup noise ("READY" etc.)
        self.ser.reset_input_buffer()

        self.HOME = (90, 120, 140)

        # [row][col] → (servo0, servo1, servo2) angles
        # Row 0 = rank 1 (a1–h1), Row 7 = rank 8 (a8–h8)
        self.chess_map = [
            [(130,72,135),(120,75,145),(110,80,150),(98,84,152),(85,82,152),(70,80,148),(55,76,145),(42,72,140)],
            [(123,67,128),(115,73,135),(105,74,140),(95,75,140),(85,75,140),(70,72,138),(60,68,138),(50,68,138)],
            [(118,60,115),(110,64,120),(100,66,128),(95,68,130),(85,67,128),(75,68,128),(65,66,125),(58,60,120)],
            [(115,55,108),(110,60,112),(100,60,118),(95,62,118),(85,63,118),(75,63,118),(68,60,115),(58,58,110)],
            [(114,43,90), (105,56,105),(99,55,105), (92,56,105),(85,55,105),(79,55,102),(70,50,100),(65,50,95) ],
            [(110,50,95), (105,45,82), (100,45,83), (90,48,90), (84,48,86), (78,48,88), (70,45,85), (65,45,80)],
            [(108,36,55), (102,40,65), (96,40,65),  (90,42,72), (84,42,72), (79,42,72), (75,40,65), (68,38,60)],
            [(108,26,30), (105,24,25), (98,26,25),  (93,32,40), (88,35,50), (80,40,50), (78,35,50), (70,32,50)],
        ]

    # -----------------------
    # SERIAL SEND
    # -----------------------
    def send(self, cmd):

        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode())

        start  = time.time()
        buffer = ""

        while True:

            if time.time() - start > 5:
                print("⚠️  Timeout waiting for Arduino")
                break

            chunk = self.ser.read(self.ser.in_waiting or 1).decode(errors="ignore")

            if not chunk:
                continue

            buffer += chunk

            if "\n" not in buffer:
                continue

            lines  = buffer.split("\n")
            buffer = lines[-1]          # keep incomplete tail

            for line in lines[:-1]:

                line = line.strip()
                if not line:
                    continue

                print("Arduino:", line)

                if "R_ON" in line or "R_OFF" in line:
                    return
                if line.startswith("S"):
                    return

    # -----------------------
    # MOTION HELPERS
    # -----------------------
    def move_normal(self, s0, s1, s2):
        """Lower arm first, then extend."""
        self.send(f"0,{s0}")
        self.send(f"2,{s2}")
        self.send(f"1,{s1}")

    def move_lift(self, s0, s1, s2):
        """Lift arm first, then rotate."""
        self.send(f"1,{s1}")
        self.send(f"2,{s2}")
        self.send(f"0,{s0}")

    # -----------------------
    # COORDINATE HELPERS
    # -----------------------
    def square_to_index(self, square):
        col = ord(square[0]) - ord('a')
        row = int(square[1]) - 1
        return row, col

    def get_angles(self, square):
        r, c = self.square_to_index(square)
        return self.chess_map[r][c]

    # -----------------------
    # HIGH-LEVEL MOTION
    # -----------------------
    def go_home(self):
        self.move_lift(*self.HOME)

    def move_to(self, square):
        self.move_normal(*self.get_angles(square))

    def magnet_on(self):
        self.send("on")

    def magnet_off(self):
        self.send("off")

    # -----------------------
    # CHESS ACTIONS
    # -----------------------
    def pick(self, square):
        print("Picking:", square)
        self.move_to(square)
        self.magnet_on()
        self.go_home()

    def place(self, square):
        print("Placing:", square)
        self.move_to(square)
        self.magnet_off()
        self.go_home()

    def execute_move(self, from_sq, to_sq):
        print(f"Executing: {from_sq} → {to_sq}")
        self.pick(from_sq)
        self.place(to_sq)

    def capture_piece(self, square, dump_square="h8"):
        print("Capturing:", square)
        self.pick(square)
        self.place(dump_square)

    # -----------------------
    # CLEANUP
    # -----------------------
    def close(self):
        self.ser.close()


# ══════════════════════════════════════════════════════════════
#  PORT DETECTION
# ══════════════════════════════════════════════════════════════

def find_arduino_port():

    for port in serial.tools.list_ports.comports():
        if (
            "Arduino"  in port.description
            or "ttyACM" in port.device
            or "ttyUSB" in port.device
        ):
            print(f"Auto-detected Arduino on {port.device}")
            return port.device

    return None


def choose_port():

    ports = list(serial.tools.list_ports.comports())

    if not ports:
        raise RuntimeError("No serial ports found. Is the Arduino connected?")

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device}  ({port.description})")

    idx = int(input("Select port number: "))
    return ports[idx].device


# ══════════════════════════════════════════════════════════════
#  MAIN GAME LOOP
# ══════════════════════════════════════════════════════════════

def main():

    # ---------- initialise subsystems ----------
    vision = VisionSystem()

    port = find_arduino_port() or choose_port()
    robot  = Manipulator(port=port)
    engine = ChessEngine(ENGINE_PATH)
    ui     = ChessBoardUI(engine)

    # ---------- calibrate camera ----------
    vision.calibrate()

    print("\n" + "═" * 40)
    print("  System Ready")
    print("  Robot = WHITE   |   You = BLACK")
    print("  Press 'd' to capture a frame")
    print("  Press ESC to quit")
    print("═" * 40 + "\n")

    turn           = "engine"   # engine (White) moves first
    awaiting_after = False
    before         = None

    # ─────────────────────────────────────────
    while True:

        # ── UI ──────────────────────────────
        if not ui.handle_events():
            break
        ui.update()

        # ── Camera view ─────────────────────
        board_frame, _ = vision.get_board_frame()
        if board_frame is not None:
            cv2.imshow("board", board_frame)

        key = cv2.waitKey(1)

        # ════════════════════════════════════
        #  ENGINE TURN  (White / Robot)
        # ════════════════════════════════════
        if turn == "engine":

            print("\nEngine (White) thinking…")

            move             = engine.get_best_move()
            from_sq, to_sq   = engine.move_to_squares(move)

            print(f"Engine: {from_sq} → {to_sq}")

            # Remove captured piece first
            if engine.is_capture(move):
                robot.capture_piece(to_sq)

            # BEFORE snapshot
            print("Press 'd' BEFORE the robot moves…")
            while True:
                if cv2.waitKey(1) == ord('d'):
                    before = vision.capture_frame()
                    print("  ✓ BEFORE captured")
                    break
                time.sleep(0.01)

            robot.execute_move(from_sq, to_sq)
            time.sleep(2)

            # AFTER snapshot
            print("Press 'd' AFTER the robot finishes…")
            while True:
                if cv2.waitKey(1) == ord('d'):
                    after = vision.capture_frame()
                    print("  ✓ AFTER captured")
                    break
                time.sleep(0.01)

            # Verify (informational only — we always apply the move)
            verified = vision.verify_move(before, after, from_sq, to_sq)
            if not verified:
                print("⚠️  Vision verification failed — forcing sync")

            engine.apply_engine_move(move)
            engine.print_pretty_board()

            if engine.is_game_over():
                print(f"\n🏁 Game over: {engine.get_result()}")
                break

            turn = "human"

        # ════════════════════════════════════
        #  HUMAN TURN  (Black)
        # ════════════════════════════════════
        if key == ord('d') and turn == "human":

            # First press → BEFORE snapshot
            if not awaiting_after:
                before         = vision.capture_frame()
                awaiting_after = True
                print("\n  ✓ BEFORE captured — make your move, then press 'd' again")

            # Second press → AFTER snapshot + detect
            else:
                after = vision.capture_frame()
                from_sq, to_sq = vision.detect_move(before, after)

                if from_sq is None:
                    print("⚠️  Could not detect move. Try again (press 'd' twice).")
                    awaiting_after = False
                    continue

                print(f"Human (Black): {from_sq} → {to_sq}")

                if not engine.apply_move_uci(from_sq, to_sq):
                    print("⚠️  Illegal move. Try again.")
                    awaiting_after = False
                    continue

                engine.print_pretty_board()

                if engine.is_game_over():
                    print(f"\n🏁 Game over: {engine.get_result()}")
                    break

                awaiting_after = False
                turn           = "engine"

        # ESC → quit
        if key == 27:
            break

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────
    engine.close()
    robot.close()
    vision.release()
    pygame.quit()
    print("Goodbye 👋")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()