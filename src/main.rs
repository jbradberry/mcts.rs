

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
    x: u8,
    y: u8
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
        let value = 1 << (action.y * 8 + action.x);
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
        if action.x >= 8 { return false }
        if action.y >= 8 { return false }

        let occupied = self.pawn1 | self.pawn2 | self.stones1 | self.stones2;
        let value = 1 << (action.y * 8 + action.x);

        if value & occupied != 0 { return false }

        match action {
            ChongAction { piece: ChongPiece::Pawn, .. } =>
                true,
            ChongAction { piece: ChongPiece::Stone, .. } => {
                if action.y == 0 || action.y == 7 { false }
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


fn main() {
    let start = ChongState::starting_state();
    println!("{:?}", start);
    println!("{:?}", start.current_player());
    println!("{:?}", start.previous_player());
    let action = ChongAction { piece: ChongPiece::Pawn, x: 1, y: 3 };
    println!("{:?}", start.next_state(&action))
}
