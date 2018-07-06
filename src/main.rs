use std::str;


pub trait BoardPlayer {}


pub trait BoardAction {}


pub trait BoardState<A: BoardAction, P: BoardPlayer> {
    fn starting_state() -> Self;

    fn previous_player(&self) -> P;

    fn current_player(&self) -> P;

    fn next_state(&self, action: &A) -> Self;

    fn is_legal(&self, action: &A, history: &[Self]) -> bool where Self: Sized;

    // fn legal_actions(&self) -> Vec<T>;

    // fn is_ended(&self) -> bool;
}


#[derive(Debug, Clone, Copy)]
pub enum ChongPlayer {
    Player1,
    Player2
}


#[derive(Debug)]
pub struct ChongState {
    pawn1: u64,
    pawn2: u64,
    stones1: u64,
    stones2: u64,
    next: ChongPlayer
}


#[derive(Debug)]
pub enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug)]
pub struct ChongAction {
    piece: ChongPiece,
    r: u8,
    c: u8
}


impl BoardPlayer for ChongPlayer {}


impl ChongPlayer {
    fn next_player(&self) -> ChongPlayer {
        match self {
            ChongPlayer::Player1 => ChongPlayer::Player2,
            ChongPlayer::Player2 => ChongPlayer::Player1
        }
    }
}


impl BoardAction for ChongAction {}


impl BoardState<ChongAction, ChongPlayer> for ChongState {
    fn starting_state() -> Self {
        Self {
            pawn1: 1 << (0 * 8 + 3),
            pawn2: 1 << (7 * 8 + 4),
            stones1: 0,
            stones2: 0,
            next: ChongPlayer::Player1
        }
    }

    fn previous_player(&self) -> ChongPlayer {
        self.next.next_player()
    }

    fn current_player(&self) -> ChongPlayer {
        self.next
    }

    fn next_state(&self, action: &ChongAction) -> Self {
        let player = self.current_player();
        let value = 1 << (action.r * 8 + action.c);
        match action {
            ChongAction { piece: ChongPiece::Pawn, .. } =>
                match player {
                    ChongPlayer::Player1 =>
                        Self { pawn1: value, next: self.next.next_player(), ..*self },
                    ChongPlayer::Player2 =>
                        Self { pawn2: value, next: self.next.next_player(), ..*self },
                }
            ChongAction { piece: ChongPiece::Stone, .. } =>
                match player {
                    ChongPlayer::Player1 =>
                        Self { stones1: value, next: self.next.next_player(), ..*self },
                    ChongPlayer::Player2 =>
                        Self { stones2: value, next: self.next.next_player(), ..*self },
                }
        }
    }

    fn is_legal(&self, action: &ChongAction, history: &[ChongState]) -> bool {
        if action.r >= 8 { return false }
        if action.c >= 8 { return false }

        let occupied = self.pawn1 | self.pawn2 | self.stones1 | self.stones2;
        let value = 1 << (action.r * 8 + action.c);

        if value & occupied != 0 { return false }

        match action {
            ChongAction { piece: ChongPiece::Pawn, .. } => {
                let player = self.current_player();
                let (pawn, stones) = match player {
                    ChongPlayer::Player1 => (self.pawn1, self.stones1),
                    ChongPlayer::Player2 => (self.pawn2, self.stones2)
                };

                // println!("{}, {}", pawn, value);
                if (pawn << 8) == value || (pawn >> 8) == value { true }
                else if ((pawn << 1) & 0xfefefefefefefefe) == value { true }
                else if ((pawn >> 1) & 0x7f7f7f7f7f7f7f7f) == value { true }
                else if (pawn << 16) == value && ((pawn << 8) & stones) != 0 { true }
                else if (pawn >> 16) == value && ((pawn >> 8) & stones) != 0 { true }
                else if ((pawn << 2) & 0xfefefefefefefefe) == value && ((pawn << 1) & stones) != 0 { true }
                else if ((pawn >> 2) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 1) & stones) != 0 { true }
                else if ((pawn << 14) & 0xfefefefefefefefe) == value && ((pawn << 7) & stones) != 0 { true }
                else if ((pawn >> 14) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 7) & stones) != 0 { true }
                else if ((pawn << 18) & 0xfefefefefefefefe) == value && ((pawn << 9) & stones) != 0 { true }
                else if ((pawn >> 18) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 9) & stones) != 0 { true }
                else { false }
            },
            ChongAction { piece: ChongPiece::Stone, .. } => {
                if action.r == 0 || action.r == 7 { false }
                else if self.stones_remaining(self.next) == 0 { false }
                else { true }
            }
        }
    }
}


impl ChongState {
    fn stones_remaining(&self, player: ChongPlayer) -> u32 {
        match player {
            ChongPlayer::Player1 => 6 - self.stones1.count_ones(),
            ChongPlayer::Player2 => 7 - self.stones2.count_ones(),
        }
    }
}


fn test_pawns(position: &ChongState) -> String {
    let mut output = [[' ' as u8; 9]; 8];
    for r in 0..8 {
        output[r as usize][8] = '\n' as u8;
        for c in 0..8 {
            let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
            output[r as usize][c as usize] = match position.is_legal(&action, &[]) {
                true => '*' as u8,
                false => '.' as u8
            }
        }
    }

    output.iter()
        .map(|x| str::from_utf8(x).unwrap())
        .collect()
}


fn main() {
    let position = ChongState { pawn1: 1 << (0 * 8 + 3), pawn2: 0,
                                stones1: 0, stones2: 0, next: ChongPlayer::Player1 };
    println!("{}\n", test_pawns(&position));

    let position = ChongState { pawn1: 1 << (3 * 8 + 5), pawn2: 0,
                                stones1: 0, stones2: 0, next: ChongPlayer::Player1 };
    println!("{}\n", test_pawns(&position));

    let position = ChongState { pawn1: 1 << (7 * 8 + 4), pawn2: 0,
                                stones1: 0, stones2: 0, next: ChongPlayer::Player1 };
    println!("{}\n", test_pawns(&position));

    let position = ChongState { pawn1: 1 << (5 * 8 + 0), pawn2: 0,
                                stones1: 0, stones2: 0, next: ChongPlayer::Player1 };
    println!("{}\n", test_pawns(&position));

    let position = ChongState { pawn1: 1 << (4 * 8 + 7), pawn2: 0,
                                stones1: 0, stones2: 0, next: ChongPlayer::Player1 };
    println!("{}\n", test_pawns(&position));

    // println!("{:?}", start.is_legal(&action, &[]));
    // println!("{:?}", start.next_state(&action));
}
