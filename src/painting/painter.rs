use crate::terminal_extensions::semantic_prompt::{PromptKind, SemanticPromptMarkers};
#[cfg(feature = "helix")]
use crate::PromptHelixMode;
use crate::{CursorConfig, PromptEditMode, PromptViMode};

#[cfg(feature = "external_printer")]
use crate::LineBuffer;
use {
    super::utils::{
        advance_printed_cursor_position, coerce_crlf, estimate_required_lines, line_width,
        PrintedCursorPosition,
    },
    crate::{
        menu::{Menu, ReedlineMenu},
        painting::PromptLines,
        Prompt,
    },
    crossterm::{
        cursor::{self, MoveLeft, MoveRight, MoveTo, MoveUp},
        style::{Attribute, Print, ResetColor, SetAttribute},
        terminal::{self, Clear, ClearType},
        QueueableCommand,
    },
    std::io::{Result, Write},
    std::ops::RangeInclusive,
    unicode_segmentation::UnicodeSegmentation,
    unicode_width::UnicodeWidthStr,
};

// Returns a string that skips N number of lines with the next offset of lines
// An offset of 0 would return only one line after skipping the required lines
fn skip_buffer_lines(string: &str, skip: usize, offset: Option<usize>) -> &str {
    let mut matches = string.match_indices('\n');
    let index = if skip == 0 {
        0
    } else {
        matches
            .clone()
            .nth(skip - 1)
            .map(|(index, _)| index + 1)
            .unwrap_or(string.len())
    };

    let limit = match offset {
        Some(offset) => {
            let offset = skip + offset;
            matches
                .nth(offset)
                .map(|(index, _)| index)
                .unwrap_or(string.len())
        }
        None => string.len(),
    };

    string[index..limit].trim_end_matches(['\r', '\n'])
}

fn skip_buffer_lines_range(string: &str, skip: usize, offset: Option<usize>) -> (usize, usize) {
    let mut matches = string.match_indices('\n');
    let index = if skip == 0 {
        0
    } else {
        matches
            .clone()
            .nth(skip - 1)
            .map(|(index, _)| index + 1)
            .unwrap_or(string.len())
    };

    let limit = match offset {
        Some(offset) => {
            let offset = skip + offset;
            matches
                .nth(offset)
                .map(|(index, _)| index)
                .unwrap_or(string.len())
        }
        None => string.len(),
    };

    (index, limit)
}

/// The writer used by crossterm operations.
///
/// In production this is a buffered stderr handle. During tests it can be
/// backed by a sink, so painting runs normally without spilling escape
/// sequences onto the real terminal: crossterm writes straight to the file
/// descriptor, which bypasses libtest's output capture.
pub enum W {
    /// Buffered stderr — the real terminal.
    // Constructed only in non-test builds; under `cfg(test)` we always use `Sink`.
    #[cfg_attr(test, allow(dead_code))]
    Terminal(std::io::BufWriter<std::io::Stderr>),
    /// Discards all output, used in tests.
    #[cfg(test)]
    Sink(std::io::Sink),
    /// Captures all output into a buffer so tests can assert on the exact
    /// escape-byte stream the painter emits (not tmux-specific — any
    /// output-level invariant).
    #[cfg(test)]
    Capture(Vec<u8>),
}

impl W {
    /// Writer targeting the real terminal (buffered stderr).
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn terminal() -> Self {
        W::Terminal(std::io::BufWriter::new(std::io::stderr()))
    }

    /// Writer that discards everything, for tests that exercise painting
    /// without printing to the terminal.
    #[cfg(test)]
    pub(crate) fn sink() -> Self {
        W::Sink(std::io::sink())
    }

    /// Writer that buffers everything written to it, so a test can inspect the
    /// emitted bytes after painting.
    #[cfg(test)]
    pub(crate) fn capture() -> Self {
        W::Capture(Vec::new())
    }

    /// Bytes captured so far. Panics unless this is a [`W::capture`] writer.
    #[cfg(test)]
    pub(crate) fn captured(&self) -> &[u8] {
        match self {
            W::Capture(buf) => buf,
            _ => panic!("captured() called on a non-capturing writer"),
        }
    }
}

impl Write for W {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        match self {
            W::Terminal(w) => w.write(buf),
            #[cfg(test)]
            W::Sink(w) => w.write(buf),
            #[cfg(test)]
            W::Capture(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self {
            W::Terminal(w) => w.flush(),
            #[cfg(test)]
            W::Sink(w) => w.flush(),
            #[cfg(test)]
            W::Capture(w) => w.flush(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct PainterSuspendedState {
    previous_prompt_rows_range: RangeInclusive<u16>,
    /// Whether the prompt reached the last row of the screen it was captured on.
    /// Recorded here rather than tested at re-use, since by then the screen may have
    /// been resized by whatever ran in between.
    was_flush_at_bottom: bool,
}

/// Screen bounds of the right prompt when it is visible.
#[derive(Debug, Clone, Copy)]
pub struct RightPromptBounds {
    pub row: u16,
    pub start_col: u16,
    pub end_col: u16,
}

#[derive(Debug, Clone)]
pub struct RenderSnapshot {
    pub screen_width: u16,
    pub screen_height: u16,
    pub prompt_start_row: u16,
    pub prompt_height: u16,
    pub large_buffer: bool,
    pub prompt_str_left: String,
    pub prompt_indicator: String,
    pub before_cursor: String,
    pub after_cursor: String,
    pub first_buffer_col: u16,
    pub menu_active: bool,
    pub menu_start_row: Option<u16>,
    pub large_buffer_extra_rows_after_prompt: Option<usize>,
    pub large_buffer_offset: Option<usize>,
    pub right_prompt: Option<RightPromptBounds>,
}

#[derive(Debug, PartialEq, Eq)]
enum PromptRowSelector {
    UseExistingPrompt { start_row: u16 },
    MakeNewPrompt { new_row: u16 },
}

// Selects the row where the next prompt should start on, taking into account and whether it should re-use a previous
// prompt.
fn select_prompt_row(
    suspended_state: Option<&PainterSuspendedState>,
    (column, row): (u16, u16), // NOTE: Positions are 0 based here
) -> PromptRowSelector {
    if let Some(painter_state) = suspended_state {
        // Re-use the previous prompt position when the cursor came back inside it,
        // unless that prompt sat flush against the bottom of the screen. A suspended
        // program that scrolled the terminal returns with the cursor pinned on the
        // bottom row, still inside the stored range and indistinguishable from an
        // in-place return, so re-using there would redraw over the scrolled-up output.
        // See nushell/reedline#1130.
        if !painter_state.was_flush_at_bottom
            && painter_state.previous_prompt_rows_range.contains(&row)
        {
            let start_row = *painter_state.previous_prompt_rows_range.start();
            return PromptRowSelector::UseExistingPrompt { start_row };
        }
    }

    // Assumption: if the cursor is not on the zeroth column,
    //   there is content we want to leave intact, thus advance to the next row.
    let new_row = if column > 0 { row + 1 } else { row };
    PromptRowSelector::MakeNewPrompt { new_row }
}

/// Layout values computed once per paint cycle, shared between rendering and snapshot creation.
pub(crate) struct PromptLayout {
    /// Total rows scrolled off the top (before prompt adjustment).
    extra_rows: usize,
    /// Rows scrolled off after subtracting prompt lines.
    extra_rows_after_prompt: usize,
    /// Lines to skip from before_cursor for menu space (large buffer only).
    large_buffer_offset: Option<usize>,

    /// Right prompt bounds (`Some` when rendered).
    right_prompt: Option<RightPromptBounds>,

    /// Row where the menu starts.
    menu_start_row: Option<u16>,

    /// Buffer start column on first visible line.
    first_buffer_col: u16,
}

/// Cached row where the prompt starts on screen, together with its
/// freshness.
///
/// Call [`PromptStartRow::invalidate`] from any new code path that
/// yields the tty or writes content the painter doesn't track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptStartRow {
    /// No row known yet (pre-`initialize_prompt_position`).
    Unverified,
    /// Last-known row, but something may have moved the cursor or written
    /// content the painter doesn't model since. The next `repaint_buffer`
    /// must re-query before trusting it.
    Stale(u16),
    /// Row matches the terminal as of the last successful query or paint.
    Verified(u16),
}

impl PromptStartRow {
    /// Painter's best understanding of the prompt's screen row,
    /// regardless of freshness. Defaults to `0` when never initialized,
    /// which should only happen before `initialize_prompt_position` runs;
    /// we don't expect that to happen in normal flow.
    pub(crate) fn last_known_row(self) -> u16 {
        match self {
            PromptStartRow::Verified(r) | PromptStartRow::Stale(r) => r,
            PromptStartRow::Unverified => 0,
        }
    }

    /// Record `row` as freshly verified.
    pub(crate) fn mark_verified(&mut self, row: u16) {
        *self = PromptStartRow::Verified(row);
    }

    /// Demote to stale, preserving the last-known row if any. Idempotent.
    pub(crate) fn invalidate(&mut self) {
        if let PromptStartRow::Verified(r) = *self {
            *self = PromptStartRow::Stale(r);
        }
    }
}

/// Whether querying the terminal for an absolute cursor position is currently
/// usable.
///
/// A tty can exist without a terminal emulator on the other end answering CPR
/// (`CSI 6 n`). Docker/Compose attach pipelines are one example. Treat that as
/// a missing terminal capability rather than as a fatal input error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorPositionSupport {
    /// No query has completed yet.
    Unknown,
    /// The most recent query succeeded.
    Supported,
    /// The most recent query failed. Normal paints must not pay the synchronous
    /// timeout again; infrequent resizes may explicitly retry.
    Unsupported,
}

impl CursorPositionSupport {
    fn should_query(self, retry_unsupported: bool) -> bool {
        self != Self::Unsupported || retry_unsupported
    }
}

/// Implementation of the output to the terminal
pub struct Painter {
    // Stdout
    stdout: W,
    prompt_start_row: PromptStartRow,
    cursor_position_support: CursorPositionSupport,
    /// Actual terminal cursor after the last paint, measured from the current
    /// prompt-local origin. Absolute CPR coordinates are used only to estimate
    /// available height and mouse bounds; ordinary rendering never targets
    /// them.
    relative_cursor_position: Option<PrintedCursorPosition>,
    // The number of lines that the prompt takes up
    prompt_height: u16,
    terminal_size: (u16, u16),
    last_required_lines: u16,
    large_buffer: bool,
    just_resized: bool,
    after_cursor_lines: Option<String>,
    /// Optional semantic prompt markers for terminal integration (OSC 133/633)
    semantic_markers: Option<Box<dyn SemanticPromptMarkers>>,
    /// Layout computed during the last paint cycle.
    pub(crate) last_layout: Option<PromptLayout>,
}

impl Painter {
    pub(crate) fn new(stdout: W) -> Self {
        Painter {
            stdout,
            prompt_start_row: PromptStartRow::Unverified,
            cursor_position_support: CursorPositionSupport::Unknown,
            relative_cursor_position: None,
            prompt_height: 0,
            terminal_size: (0, 0),
            last_required_lines: 0,
            large_buffer: false,
            just_resized: false,
            after_cursor_lines: None,
            semantic_markers: None,
            last_layout: None,
        }
    }

    /// Height of the current terminal window
    pub fn screen_height(&self) -> u16 {
        self.terminal_size.1
    }

    /// Width of the current terminal window
    pub fn screen_width(&self) -> u16 {
        self.terminal_size.0
    }

    /// Sets the semantic prompt markers for terminal integration (OSC 133/633)
    pub fn set_semantic_markers(&mut self, markers: Option<Box<dyn SemanticPromptMarkers>>) {
        self.semantic_markers = markers;
    }

    /// Returns a reference to the semantic prompt markers, if any
    pub fn semantic_markers(&self) -> Option<&dyn SemanticPromptMarkers> {
        self.semantic_markers.as_deref()
    }
    /// Returns the empty lines from the prompt down.
    pub fn remaining_lines_real(&self) -> u16 {
        self.screen_height()
            .saturating_sub(self.prompt_start_row.last_known_row())
            .saturating_sub(self.prompt_height)
    }

    /// Returns the number of lines that are available or can be made available by
    /// stripping the prompt.
    ///
    /// If you want the number of empty lines below the prompt,
    /// use [`Painter::remaining_lines_real`] instead.
    pub fn remaining_lines(&self) -> u16 {
        self.screen_height()
            .saturating_sub(self.prompt_start_row.last_known_row())
    }

    /// Computes layout values shared between rendering and snapshot creation.
    fn compute_layout(&self, lines: &PromptLines, menu: Option<&ReedlineMenu>) -> PromptLayout {
        let screen_width = self.screen_width();
        let screen_height = self.screen_height();

        // Large buffer extra rows computation
        let (extra_rows, extra_rows_after_prompt) = if self.large_buffer {
            let prompt_lines = lines.prompt_lines_with_wrap(screen_width) as usize;
            let prompt_indicator_lines = lines.prompt_indicator.lines().count();
            let before_cursor_lines = lines.before_cursor.lines().count();
            let total_lines_before =
                prompt_lines + prompt_indicator_lines + before_cursor_lines - 1;
            let extra = total_lines_before.saturating_sub(screen_height as usize);
            (extra, extra.saturating_sub(prompt_lines))
        } else {
            (0, 0)
        };

        // Large buffer offset for menu space
        let large_buffer_offset = if self.large_buffer {
            let cursor_distance = lines.distance_from_prompt(screen_width);
            menu.and_then(|menu| {
                if cursor_distance >= screen_height.saturating_sub(1) {
                    let rows = lines
                        .before_cursor
                        .lines()
                        .count()
                        .saturating_sub(extra_rows_after_prompt)
                        .saturating_sub(menu.min_rows() as usize);
                    Some(rows)
                } else {
                    None
                }
            })
        } else {
            None
        };

        // Right prompt layout — only visible when the prompt itself hasn't scrolled off
        let right_prompt =
            if lines.prompt_str_right.is_empty() || self.large_buffer && extra_rows > 0 {
                None
            } else {
                let prompt_length_right = line_width(&lines.prompt_str_right);
                let start_position = screen_width.saturating_sub(prompt_length_right as u16);
                let input_width = lines.estimate_right_prompt_line_width(screen_width);

                if input_width <= start_position {
                    let mut row = 0;
                    if lines.right_prompt_on_last_line {
                        row += lines.prompt_lines_with_wrap(screen_width);
                    }
                    Some(RightPromptBounds {
                        row,
                        start_col: start_position,
                        end_col: start_position.saturating_add(prompt_length_right as u16),
                    })
                } else {
                    None
                }
            };

        // Menu start row
        let menu_start_row = menu.map(|menu| {
            let cursor_distance = lines.distance_from_prompt(screen_width);
            if cursor_distance >= screen_height.saturating_sub(1) {
                screen_height.saturating_sub(menu.min_rows())
            } else {
                cursor_distance + 1
            }
        });

        // First buffer column
        let first_buffer_col = if self.large_buffer && extra_rows_after_prompt > 0 {
            0
        } else {
            let prompt_line = format!("{}{}", lines.prompt_str_left, lines.prompt_indicator);
            let last_prompt_line = prompt_line.lines().last().unwrap_or_default();
            let width = line_width(last_prompt_line);
            if width > u16::MAX as usize {
                u16::MAX
            } else {
                width as u16
            }
        };

        PromptLayout {
            extra_rows,
            extra_rows_after_prompt,
            large_buffer_offset,
            right_prompt,
            menu_start_row,
            first_buffer_col,
        }
    }

    /// Returns the state necessary before suspending the painter (to run a host command event).
    ///
    /// This state will be used to re-initialize the painter to re-use last prompt if possible.
    pub fn state_before_suspension(&self) -> PainterSuspendedState {
        let start_row = self.prompt_start_row.last_known_row();
        let final_row = start_row + self.last_required_lines;
        PainterSuspendedState {
            previous_prompt_rows_range: start_row..=final_row,
            // `final_row` can overshoot the last visible row for a prompt at the very
            // bottom, so this is `>=` rather than an equality.
            was_flush_at_bottom: final_row >= self.screen_height().saturating_sub(1),
        }
    }

    fn update_terminal_size(&mut self, size: (u16, u16)) {
        // Some non-compliant terminals report (0, 0). Keep the established
        // fallback so layout arithmetic never divides by zero.
        self.terminal_size = if size == (0, 0) { (80, 24) } else { size };
    }

    /// Query the cursor when the current capability state permits it.
    ///
    /// Cursor queries are advisory for painting. Other terminal I/O still
    /// returns its errors normally; only a missing CPR response is downgraded
    /// to an unavailable capability here.
    fn query_cursor_position(&mut self, retry_unsupported: bool) -> Option<(u16, u16)> {
        if !self.cursor_position_support.should_query(retry_unsupported) {
            return None;
        }

        let result = cursor::position();
        self.record_cursor_position_result(result)
    }

    fn record_cursor_position_result(&mut self, result: Result<(u16, u16)>) -> Option<(u16, u16)> {
        match result {
            Ok(position) => {
                self.cursor_position_support = CursorPositionSupport::Supported;
                Some(position)
            }
            Err(_) => {
                self.cursor_position_support = CursorPositionSupport::Unsupported;
                None
            }
        }
    }

    const fn relative_origin() -> PrintedCursorPosition {
        PrintedCursorPosition::origin()
    }

    /// Queue visible text and advance the prompt-local cursor through it.
    fn queue_tracked_print(
        &mut self,
        cursor: &mut PrintedCursorPosition,
        text: &str,
    ) -> Result<()> {
        self.stdout.queue(Print(text))?;
        *cursor = advance_printed_cursor_position(*cursor, [text], self.screen_width());
        Ok(())
    }

    /// Move between two positions measured from the same prompt origin.
    ///
    /// Moving down uses CRLF, as prompt-toolkit does, so a terminal at its
    /// bottom edge can create/scroll rows. It also gives line-oriented stream
    /// multiplexers (Docker Compose) a boundary at which to add their prefix.
    fn queue_relative_move(
        &mut self,
        current: &mut PrintedCursorPosition,
        to: PrintedCursorPosition,
    ) -> Result<()> {
        debug_assert!(
            !to.pending_wrap,
            "cursor movement cannot create deferred wrap"
        );

        let width = self.screen_width().max(1);
        let original = *current;
        let mut current_column = original.physical_column(width);
        if to.row > original.row {
            for _ in original.row..to.row {
                self.stdout.queue(Print("\r\n"))?;
            }
            current_column = 0;
        } else if to.row < original.row {
            self.stdout.queue(MoveUp(original.row - to.row))?;
        }

        if to.column < current_column {
            self.stdout.queue(MoveLeft(current_column - to.column))?;
        } else if to.column > current_column {
            self.stdout.queue(MoveRight(to.column - current_column))?;
        } else if original.pending_wrap && to.row == original.row {
            // Any cursor movement clears the terminal's deferred-wrap flag.
            // Keep the visible cell unchanged without using a carriage return,
            // which would cross a stream multiplexer prefix.
            if current_column > 0 {
                self.stdout.queue(MoveLeft(1))?.queue(MoveRight(1))?;
            } else {
                self.stdout.queue(MoveRight(1))?.queue(MoveLeft(1))?;
            }
        }
        *current = to;
        Ok(())
    }

    /// Move to a prompt-local row and erase below it without ever crossing the
    /// local origin. Starting the erase one cell to the right retains the tmux
    /// `scroll-on-clear` workaround; the skipped cell is immediately repainted.
    fn clear_from_local_row(&mut self, cursor: &mut PrintedCursorPosition, row: u16) -> Result<()> {
        let guard_column = u16::from(self.screen_width() > 1);
        self.queue_relative_move(
            cursor,
            PrintedCursorPosition {
                column: guard_column,
                row,
                pending_wrap: false,
            },
        )?;
        self.stdout.queue(Clear(ClearType::FromCursorDown))?;
        if guard_column > 0 {
            self.stdout.queue(MoveLeft(1))?;
            cursor.column = 0;
        }
        Ok(())
    }

    /// Make the cursor's current line the origin of the next local frame.
    fn adopt_cursor_line_as_origin(&mut self, cursor: PrintedCursorPosition) {
        let row = self
            .prompt_start_row
            .last_known_row()
            .saturating_add(cursor.row)
            .min(self.screen_height().saturating_sub(1));
        self.prompt_start_row.mark_verified(row);
        self.relative_cursor_position = Some(Self::relative_origin());
        self.last_required_lines = 0;
        self.last_layout = None;
        self.after_cursor_lines = None;
        self.just_resized = false;
    }

    /// Establish a prompt-local origin without assuming an absolute screen row.
    ///
    /// Stream multiplexers can inject an invisible prefix before every output
    /// line. Absolute `MoveTo(0, row)` would jump in front of that prefix and can
    /// also rewrite old log rows during replay. Relative mode owns only the
    /// cells it printed after the current line's injected prefix.
    fn establish_fallback_origin(&mut self, preserve_current_line: bool) -> Result<()> {
        if preserve_current_line {
            self.stdout.queue(Print("\r\n"))?;
        }

        self.relative_cursor_position = Some(Self::relative_origin());
        // With no CPR response, the amount of space below is unknown. Treat
        // the new local origin as bottom-aligned for conservative layout; all
        // actual movement remains relative and natural scrolling corrects it.
        self.prompt_start_row
            .mark_verified(self.screen_height().saturating_sub(1));
        self.last_required_lines = 0;
        self.just_resized = false;
        self.stdout.flush()?;
        Ok(())
    }

    fn select_and_store_prompt_position(
        &mut self,
        suspended_state: Option<&PainterSuspendedState>,
        position: (u16, u16),
    ) -> Result<()> {
        let prompt_selector = select_prompt_row(suspended_state, position);
        let (column, row) = position;
        match prompt_selector {
            PromptRowSelector::UseExistingPrompt { start_row } => {
                self.prompt_start_row.mark_verified(start_row);
                self.relative_cursor_position = Some(PrintedCursorPosition {
                    column,
                    row: row.saturating_sub(start_row),
                    pending_wrap: false,
                });
            }
            PromptRowSelector::MakeNewPrompt { new_row } => {
                let new_row = if column > 0 {
                    self.stdout.queue(Print("\r\n"))?.flush()?;
                    new_row.min(self.screen_height().saturating_sub(1))
                } else {
                    new_row
                };
                self.prompt_start_row.mark_verified(new_row);
                self.relative_cursor_position = Some(Self::relative_origin());
            }
        }
        self.last_required_lines = 0;
        self.just_resized = false;
        Ok(())
    }

    fn finish_prompt_position_initialization(
        &mut self,
        suspended_state: Option<&PainterSuspendedState>,
        position: Option<(u16, u16)>,
    ) -> Result<()> {
        match position {
            Some(position) => self.select_and_store_prompt_position(suspended_state, position),
            None => {
                // Like prompt-toolkit, a fresh renderer treats the tty's current
                // position as its local origin. Hosts normally finish any output
                // before calling `read_line`; forcing a CRLF here would therefore
                // manufacture an empty line in line-oriented logs such as Docker
                // Compose's. Keep the conservative line preservation for later
                // re-anchoring, where another program or a resize may genuinely
                // have left the cursor in an untracked position.
                let initial_prompt = suspended_state.is_none()
                    && self.relative_cursor_position.is_none()
                    && self.prompt_start_row == PromptStartRow::Unverified;
                let preserve_current_line = !initial_prompt
                    && (suspended_state.is_some()
                        || self.relative_cursor_position.is_none()
                        || matches!(self.prompt_start_row, PromptStartRow::Stale(_)));
                self.establish_fallback_origin(preserve_current_line)
            }
        }
    }

    fn reset_after_clear(&mut self, terminal_size: (u16, u16)) {
        self.update_terminal_size(terminal_size);
        self.prompt_start_row.mark_verified(0);
        self.relative_cursor_position = Some(Self::relative_origin());
        self.last_required_lines = 0;
        self.just_resized = false;
    }

    fn apply_resize_cursor_position(&mut self, position: Option<(u16, u16)>) {
        if let Some((_, row)) = position {
            self.prompt_start_row = PromptStartRow::Stale(row);
            self.just_resized = true;
        } else {
            self.relative_cursor_position = None;
        }
    }

    /// Sets the prompt origin position and screen size for a new line editor
    /// invocation
    ///
    /// Not to be used for resizes during a running line editor, use
    /// [`Painter::handle_resize()`] instead
    pub(crate) fn initialize_prompt_position(
        &mut self,
        suspended_state: Option<&PainterSuspendedState>,
    ) -> Result<()> {
        // Update the terminal size
        self.update_terminal_size(terminal::size()?);

        let position = self.query_cursor_position(false);
        self.finish_prompt_position_initialization(suspended_state, position)
    }

    /// Mark `prompt_start_row` as possibly out of sync — the next
    /// `repaint_buffer` will re-query the terminal. Call from any path
    /// that lets something other than reedline's rendering move the
    /// cursor (e.g. `$EDITOR`).
    pub(crate) fn invalidate_prompt_start_row(&mut self) {
        self.prompt_start_row.invalidate();
    }

    /// Main painter for the prompt and buffer
    /// It queues all the actions required to print the prompt together with
    /// lines that make the buffer.
    /// Using the prompt lines object in this function it is estimated how the
    /// prompt should scroll up and how much space is required to print all the
    /// lines for the buffer. All ordinary cursor motion stays inside the
    /// prompt-local frame; rows are created or scrolled naturally with CRLF.
    pub(crate) fn repaint_buffer(
        &mut self,
        prompt: &dyn Prompt,
        lines: &PromptLines,
        prompt_mode: PromptEditMode,
        menu: Option<&ReedlineMenu>,
        use_ansi_coloring: bool,
        cursor_config: &Option<CursorConfig>,
    ) -> Result<()> {
        // Reset any ANSI styling that may have been left by external commands
        // This ensures the prompt is not affected by previous output styling
        // Note: Attribute::Reset (SGR 0) resets all attributes including colors
        self.stdout.queue(SetAttribute(Attribute::Reset))?;

        self.stdout.queue(cursor::Hide)?;

        let screen_width = self.screen_width();
        let screen_height = self.screen_height();

        // We add one here as [`PromptLines::prompt_lines_with_wrap`] intentionally subtracts 1 from the real value.
        self.prompt_height = lines.prompt_lines_with_wrap(screen_width) + 1;
        let required_lines = lines.required_lines(screen_width, false, menu);

        // Marking the painter state as larger buffer to avoid animations.
        self.large_buffer = required_lines >= screen_height;

        // A supported terminal reports where reflow left the cursor after a
        // resize. Convert that measurement back into this renderer's local
        // coordinates; CPR remains advisory and is never used as a paint target.
        if self.just_resized {
            let measured_cursor_row = self.prompt_start_row.last_known_row();
            let local_cursor = advance_printed_cursor_position(
                Self::relative_origin(),
                [
                    &*lines.prompt_str_left,
                    &lines.prompt_indicator,
                    &lines.before_cursor,
                ],
                screen_width,
            );
            if local_cursor.row < screen_height {
                self.prompt_start_row
                    .mark_verified(measured_cursor_row.saturating_sub(local_cursor.row));
                self.relative_cursor_position = Some(local_cursor);
            } else {
                self.relative_cursor_position = None;
            }
            self.just_resized = false;
        }

        // Reconcile a stale anchor: something yielded the tty since the last paint
        // (a resize, an external completer, `$EDITOR`) and may have scrolled our
        // content.
        if let PromptStartRow::Stale(row) = self.prompt_start_row {
            // Cursor above the cached row => content scrolled up while the tty
            // was yielded. Re-anchor to the cursor (ground truth) rather than
            // homing to row 0, which would yank the prompt to the top. The `+1`
            // allows for output that left the cursor on the prompt row.
            // See nushell/reedline#1130.
            match self.query_cursor_position(false) {
                Some((cursor_column, cursor_row)) => {
                    let anchor = if cursor_row + 1 < row {
                        cursor_row
                    } else {
                        row
                    };
                    self.prompt_start_row.mark_verified(anchor);
                    self.relative_cursor_position = Some(PrintedCursorPosition {
                        column: cursor_column,
                        row: cursor_row.saturating_sub(anchor),
                        pending_wrap: false,
                    });
                }
                None => self.establish_fallback_origin(true)?,
            }
        }

        if self.relative_cursor_position.is_none()
            || self.prompt_start_row == PromptStartRow::Unverified
        {
            self.establish_fallback_origin(true)?;
        }

        let old_anchor_row = self.prompt_start_row.last_known_row();
        let mut terminal_cursor = self
            .relative_cursor_position
            .unwrap_or_else(Self::relative_origin);
        self.clear_from_local_row(&mut terminal_cursor, 0)?;

        let layout = self.compute_layout(lines, menu);

        let edit_cursor = if self.large_buffer {
            self.print_large_buffer(
                prompt,
                lines,
                menu,
                use_ansi_coloring,
                &layout,
                &mut terminal_cursor,
            )?
        } else {
            self.print_small_buffer(
                prompt,
                lines,
                menu,
                use_ansi_coloring,
                &layout,
                &mut terminal_cursor,
            )?
        };

        debug_assert_eq!(terminal_cursor, edit_cursor);
        self.relative_cursor_position = Some(edit_cursor);

        self.last_layout = Some(layout);

        // The last_required_lines is used to calculate safe range of the current prompt.
        self.last_required_lines = required_lines;

        self.after_cursor_lines = if !lines.after_cursor.is_empty() {
            Some(lines.after_cursor.to_string())
        } else {
            None
        };

        // Printing below the current physical bottom scrolls the complete local
        // frame upward. Keep the absolute row only as advisory layout/mouse
        // metadata; no output operation targets it.
        let visible_rows = required_lines.max(1).min(screen_height);
        let scrolled = old_anchor_row
            .saturating_add(visible_rows)
            .saturating_sub(screen_height);
        self.prompt_start_row
            .mark_verified(old_anchor_row.saturating_sub(scrolled));

        if let Some(shapes) = cursor_config {
            let shape = match &prompt_mode {
                PromptEditMode::Emacs => shapes.emacs,
                PromptEditMode::Vi(PromptViMode::Insert) => shapes.vi_insert,
                PromptEditMode::Vi(PromptViMode::Normal | PromptViMode::Visual) => shapes.vi_normal,
                #[cfg(feature = "helix")]
                PromptEditMode::Helix(PromptHelixMode::Insert) => shapes.hx_insert,
                #[cfg(feature = "helix")]
                PromptEditMode::Helix(PromptHelixMode::Normal) => shapes.hx_normal,
                #[cfg(feature = "helix")]
                PromptEditMode::Helix(PromptHelixMode::Select) => shapes.hx_select,
                _ => None,
            };
            if let Some(shape) = shape {
                self.stdout.queue(shape)?;
            }
        }
        self.stdout.queue(cursor::Show)?;

        self.stdout.flush()
    }

    /// Captures the current screen layout into a [`RenderSnapshot`] that records
    /// prompt geometry, buffer positions, right-prompt bounds, and menu state.
    /// This snapshot is later used by [`Self::screen_to_buffer_offset`] to map a
    /// terminal (column, row) click coordinate to a byte offset in the editing buffer.
    pub(crate) fn render_snapshot(
        &self,
        lines: &PromptLines,
        menu: Option<&ReedlineMenu>,
        raw_before: &str,
        raw_after: &str,
        layout: &PromptLayout,
    ) -> RenderSnapshot {
        let large_buffer_extra_rows_after_prompt = if self.large_buffer {
            Some(layout.extra_rows_after_prompt)
        } else {
            None
        };
        let large_buffer_offset = layout.large_buffer_offset;

        let prompt_start_row = self.prompt_start_row.last_known_row();
        let right_prompt = layout.right_prompt.map(|bounds| RightPromptBounds {
            row: prompt_start_row.saturating_add(bounds.row),
            ..bounds
        });
        let menu_start_row = layout
            .menu_start_row
            .map(|row| prompt_start_row.saturating_add(row));

        RenderSnapshot {
            screen_width: self.screen_width(),
            screen_height: self.screen_height(),
            prompt_start_row,
            prompt_height: self.prompt_height,
            large_buffer: self.large_buffer,
            prompt_str_left: lines.prompt_str_left.to_string(),
            prompt_indicator: lines.prompt_indicator.to_string(),
            before_cursor: raw_before.to_string(),
            after_cursor: raw_after.to_string(),
            first_buffer_col: layout.first_buffer_col,
            menu_active: menu.is_some(),
            menu_start_row,
            large_buffer_extra_rows_after_prompt,
            large_buffer_offset,
            right_prompt,
        }
    }

    /// Maps a terminal screen coordinate (column, row) to a byte offset in the
    /// combined editing buffer (`before_cursor + after_cursor`).
    ///
    /// Returns `None` when the click lands outside the editable area: above the
    /// prompt, inside the right prompt, inside the menu, or past the end of
    /// visible buffer content.
    ///
    /// The algorithm walks grapheme-by-grapheme through the visible portion of
    /// the buffer, tracking the current (row, col) on screen. Wide characters
    /// and line wrapping are accounted for. When the tracked position matches
    /// the target coordinate, the corresponding byte offset is returned.
    pub(crate) fn screen_to_buffer_offset(
        &self,
        snapshot: &RenderSnapshot,
        column: u16,
        row: u16,
    ) -> Option<usize> {
        // Clicks above the prompt are not in the buffer.
        if row < snapshot.prompt_start_row {
            return None;
        }

        // Clicks inside the menu area are not in the buffer.
        if snapshot.menu_active {
            if let Some(menu_start_row) = snapshot.menu_start_row {
                if row >= menu_start_row {
                    return None;
                }
            }
        }

        // Clicks inside the right prompt area are not in the buffer.
        if let Some(rp) = &snapshot.right_prompt {
            if row == rp.row && column >= rp.start_col && column < rp.end_col {
                return None;
            }
        }

        // Convert the absolute screen row to a row relative to the prompt start.
        let screen_width = snapshot.screen_width;
        let target_row = row.saturating_sub(snapshot.prompt_start_row);

        // Determine which relative row the buffer content begins on. When the
        // buffer hasn't scrolled, it starts on the last line of the prompt;
        // otherwise it starts at row 0 (the prompt itself has scrolled off).
        let buffer_start_row = if snapshot.large_buffer
            && snapshot.large_buffer_extra_rows_after_prompt.unwrap_or(0) > 0
        {
            0
        } else {
            snapshot.prompt_height.saturating_sub(1)
        };

        // Click landed in the prompt area before any buffer text.
        if target_row < buffer_start_row {
            return None;
        }

        // Compute the visible byte ranges of the before-cursor and after-cursor
        // buffer segments, accounting for lines scrolled off-screen in large
        // buffers and space reserved for menus.
        let (before_start, before_end) = if snapshot.large_buffer {
            skip_buffer_lines_range(
                &snapshot.before_cursor,
                snapshot.large_buffer_extra_rows_after_prompt.unwrap_or(0),
                snapshot.large_buffer_offset,
            )
        } else {
            (0, snapshot.before_cursor.len())
        };
        let before_visible = &snapshot.before_cursor[before_start..before_end];
        let full_before_visible = before_start == 0 && before_end == snapshot.before_cursor.len();

        let (after_start, after_end) = if snapshot.large_buffer {
            if snapshot.menu_active {
                let end = snapshot
                    .after_cursor
                    .find('\n')
                    .unwrap_or(snapshot.after_cursor.len());
                (0, end)
            } else {
                let cursor_distance = estimate_required_lines(
                    &format!(
                        "{}{}{}",
                        snapshot.prompt_str_left, snapshot.prompt_indicator, snapshot.before_cursor
                    ),
                    screen_width,
                )
                .saturating_sub(1) as u16;
                let remaining_lines = snapshot.screen_height.saturating_sub(cursor_distance);
                let offset = remaining_lines.saturating_sub(1) as usize;
                skip_buffer_lines_range(&snapshot.after_cursor, 0, Some(offset))
            }
        } else {
            (0, snapshot.after_cursor.len())
        };
        let after_visible = &snapshot.after_cursor[after_start..after_end];
        let full_after_visible = after_start == 0 && after_end == snapshot.after_cursor.len();
        let full_buffer_visible = full_before_visible && full_after_visible;

        // Walk through visible buffer content grapheme-by-grapheme, tracking
        // the screen position. When we hit the target (column, row) we return
        // the corresponding byte offset in the full buffer.
        let mut current_row = buffer_start_row;
        let mut current_col = if current_row == buffer_start_row {
            snapshot.first_buffer_col
        } else {
            0
        };

        let mut check_segment = |segment: &str, base_offset: usize| -> Option<usize> {
            for (index, grapheme) in segment.grapheme_indices(true) {
                if grapheme == "\n" {
                    current_row = current_row.saturating_add(1);
                    current_col = 0;
                    continue;
                }

                let width = grapheme.width().max(1) as u16;
                if current_col.saturating_add(width) > screen_width {
                    current_row = current_row.saturating_add(1);
                    current_col = 0;
                }

                if current_row == target_row
                    && column >= current_col
                    && column < current_col.saturating_add(width)
                {
                    return Some(base_offset + index);
                }

                current_col = current_col.saturating_add(width);
            }

            None
        };

        if let Some(offset) = check_segment(before_visible, before_start) {
            return Some(offset);
        }

        let after_base = snapshot.before_cursor.len().saturating_add(after_start);
        if let Some(offset) = check_segment(after_visible, after_base) {
            return Some(offset);
        }

        // Click is past all buffer content but still on the last buffer row;
        // treat it as a click at the very end of the buffer.
        if full_buffer_visible && target_row == current_row && column >= current_col {
            return Some(snapshot.before_cursor.len() + snapshot.after_cursor.len());
        }

        None
    }

    /// Paint the right prompt at a prompt-local position and return to the
    /// cursor that preceded it. If that cursor was in deferred-wrap state, no
    /// cursor command can recreate the flag; return `true` after moving to the
    /// origin so the caller can repaint its path naturally.
    fn print_right_prompt(
        &mut self,
        lines: &PromptLines,
        layout: &PromptLayout,
        cursor: &mut PrintedCursorPosition,
    ) -> Result<bool> {
        let Some(rp) = &layout.right_prompt else {
            return Ok(false);
        };
        let return_position = *cursor;
        self.queue_relative_move(
            cursor,
            PrintedCursorPosition {
                column: rp.start_col,
                row: rp.row,
                pending_wrap: false,
            },
        )?;

        // Emit right prompt marker (OSC 133;P;k=r)
        if let Some(markers) = &self.semantic_markers {
            self.stdout
                .queue(Print(markers.prompt_start(PromptKind::Right)))?;
        }

        let right = coerce_crlf(&lines.prompt_str_right);
        self.queue_tracked_print(cursor, &right)?;

        if return_position.pending_wrap {
            self.queue_relative_move(cursor, Self::relative_origin())?;
            Ok(true)
        } else {
            self.queue_relative_move(cursor, return_position)?;
            Ok(false)
        }
    }

    fn print_menu(
        &mut self,
        menu: &dyn Menu,
        use_ansi_coloring: bool,
        layout: &PromptLayout,
        cursor: &mut PrintedCursorPosition,
    ) -> Result<()> {
        let starting_row = layout.menu_start_row.unwrap_or(0);
        let remaining_lines = self.screen_height().saturating_sub(starting_row);
        let menu_string = menu.menu_string(remaining_lines, use_ansi_coloring);
        self.clear_from_local_row(cursor, starting_row)?;
        let menu_string = coerce_crlf(menu_string.trim_end_matches('\n'));
        self.queue_tracked_print(cursor, &menu_string)?;

        Ok(())
    }

    /// Repaint the visible path from the local origin to the edit cursor. This
    /// is needed only when that cursor sits in deferred-wrap state: relative
    /// cursor commands can reach the same cell, but cannot recreate the flag
    /// that makes the next glyph wrap.
    fn reprint_cursor_path(
        &mut self,
        prompt: &dyn Prompt,
        prompt_left: &str,
        indicator: &str,
        before_cursor: &str,
        use_ansi_coloring: bool,
        cursor: &mut PrintedCursorPosition,
    ) -> Result<()> {
        debug_assert_eq!(*cursor, Self::relative_origin());
        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_prompt_color().prefix()))?;
        }
        let prompt_left = coerce_crlf(prompt_left);
        self.queue_tracked_print(cursor, &prompt_left)?;

        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_indicator_color().prefix()))?;
        }
        let indicator = coerce_crlf(indicator);
        self.queue_tracked_print(cursor, &indicator)?;

        if use_ansi_coloring {
            self.stdout
                .queue(SetAttribute(Attribute::Reset))?
                .queue(ResetColor)?;
        }
        self.queue_tracked_print(cursor, before_cursor)
    }

    fn print_small_buffer(
        &mut self,
        prompt: &dyn Prompt,
        lines: &PromptLines,
        menu: Option<&ReedlineMenu>,
        use_ansi_coloring: bool,
        layout: &PromptLayout,
        cursor: &mut PrintedCursorPosition,
    ) -> Result<PrintedCursorPosition> {
        // Emit prompt start marker (OSC 133;A;k=i for primary prompt)
        if let Some(markers) = &self.semantic_markers {
            self.stdout
                .queue(Print(markers.prompt_start(PromptKind::Primary)))?;
        }

        // print our prompt with color
        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_prompt_color().prefix()))?;
        }

        let prompt_left = coerce_crlf(&lines.prompt_str_left);
        self.queue_tracked_print(cursor, &prompt_left)?;

        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_indicator_color().prefix()))?;
        }

        let indicator = coerce_crlf(&lines.prompt_indicator);
        self.queue_tracked_print(cursor, &indicator)?;

        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_prompt_right_color().prefix()))?;
        }

        let repaint_prompt = self.print_right_prompt(lines, layout, cursor)?;
        if repaint_prompt {
            self.reprint_cursor_path(
                prompt,
                &lines.prompt_str_left,
                &lines.prompt_indicator,
                "",
                use_ansi_coloring,
                cursor,
            )?;
        }

        // Emit command input start marker (OSC 133;B) after prompt (including right prompt)
        if let Some(markers) = &self.semantic_markers {
            self.stdout.queue(Print(markers.command_input_start()))?;
        }

        if use_ansi_coloring {
            self.stdout
                .queue(SetAttribute(Attribute::Reset))?
                .queue(ResetColor)?;
        }

        self.queue_tracked_print(cursor, &lines.before_cursor)?;
        let edit_cursor = *cursor;
        self.queue_tracked_print(cursor, &lines.after_cursor)?;

        if let Some(menu) = menu {
            self.print_menu(menu, use_ansi_coloring, layout, cursor)?;
        } else {
            self.queue_tracked_print(cursor, &lines.hint)?;
        }

        if *cursor != edit_cursor {
            if edit_cursor.pending_wrap {
                self.queue_relative_move(cursor, Self::relative_origin())?;
                self.reprint_cursor_path(
                    prompt,
                    &lines.prompt_str_left,
                    &lines.prompt_indicator,
                    &lines.before_cursor,
                    use_ansi_coloring,
                    cursor,
                )?;
            } else {
                self.queue_relative_move(cursor, edit_cursor)?;
            }
        }

        Ok(edit_cursor)
    }

    fn print_large_buffer(
        &mut self,
        prompt: &dyn Prompt,
        lines: &PromptLines,
        menu: Option<&ReedlineMenu>,
        use_ansi_coloring: bool,
        layout: &PromptLayout,
        cursor: &mut PrintedCursorPosition,
    ) -> Result<PrintedCursorPosition> {
        let screen_width = self.screen_width();
        let screen_height = self.screen_height();
        let cursor_distance = lines.distance_from_prompt(screen_width);
        let remaining_lines = screen_height.saturating_sub(cursor_distance);

        let extra_rows = layout.extra_rows;
        let extra_rows_after_prompt = layout.extra_rows_after_prompt;

        // Emit prompt start marker (OSC 133;A;k=i for primary prompt) only if prompt is visible
        if extra_rows == 0 {
            if let Some(markers) = &self.semantic_markers {
                self.stdout
                    .queue(Print(markers.prompt_start(PromptKind::Primary)))?;
            }
        }

        // print our prompt with color
        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_prompt_color().prefix()))?;
        }

        // In case the prompt is made out of multiple lines, the prompt is split by
        // lines and only the required ones are printed.
        //
        // Sliced rather than run through `skip_buffer_lines`, whose trailing-newline
        // trim would drop the newline that puts the indicator on the row below.
        let (prompt_start, prompt_end) =
            skip_buffer_lines_range(&lines.prompt_str_left, extra_rows, None);
        let prompt_skipped = &lines.prompt_str_left[prompt_start..prompt_end];
        let prompt_text = coerce_crlf(prompt_skipped);
        self.queue_tracked_print(cursor, &prompt_text)?;

        if extra_rows == 0 {
            if use_ansi_coloring {
                self.stdout
                    .queue(Print(prompt.get_prompt_right_color().prefix()))?;
            }

            let repaint_prompt = self.print_right_prompt(lines, layout, cursor)?;
            if repaint_prompt {
                self.reprint_cursor_path(
                    prompt,
                    prompt_skipped,
                    "",
                    "",
                    use_ansi_coloring,
                    cursor,
                )?;
            }
        }

        if use_ansi_coloring {
            self.stdout
                .queue(Print(prompt.get_indicator_color().prefix()))?;
        }
        let indicator_skipped =
            skip_buffer_lines(&lines.prompt_indicator, extra_rows_after_prompt, None);
        let indicator_text = coerce_crlf(indicator_skipped);
        self.queue_tracked_print(cursor, &indicator_text)?;

        // Emit command input start marker (OSC 133;B) after prompt indicator
        if let Some(markers) = &self.semantic_markers {
            self.stdout.queue(Print(markers.command_input_start()))?;
        }

        if use_ansi_coloring {
            self.stdout.queue(ResetColor)?;
        }

        // Selecting the lines before the cursor that will be printed
        let before_cursor_skipped = skip_buffer_lines(
            &lines.before_cursor,
            extra_rows_after_prompt,
            layout.large_buffer_offset,
        );
        self.queue_tracked_print(cursor, before_cursor_skipped)?;
        let edit_cursor = *cursor;

        if let Some(menu) = menu {
            // TODO: Also solve the difficult problem of displaying (parts of)
            // the content after the cursor with the completion menu
            // This only shows the rest of the line the cursor is on
            if let Some(newline) = lines.after_cursor.find('\n') {
                self.queue_tracked_print(cursor, &lines.after_cursor[0..newline])?;
            } else {
                self.queue_tracked_print(cursor, &lines.after_cursor)?;
            }
            self.print_menu(menu, use_ansi_coloring, layout, cursor)?;
        } else {
            // Selecting lines for the hint
            // The -1 subtraction is done because the remaining lines consider the line where the
            // cursor is located as a remaining line. That has to be removed to get the correct offset
            // for the after-cursor and hint lines
            let offset = remaining_lines.saturating_sub(1) as usize;
            // Selecting lines after the cursor
            let after_cursor_skipped = skip_buffer_lines(&lines.after_cursor, 0, Some(offset));
            self.queue_tracked_print(cursor, after_cursor_skipped)?;
            // Hint lines
            let hint_skipped = skip_buffer_lines(&lines.hint, 0, Some(offset));
            self.queue_tracked_print(cursor, hint_skipped)?;
        }

        if *cursor != edit_cursor {
            if edit_cursor.pending_wrap {
                self.queue_relative_move(cursor, Self::relative_origin())?;
                self.reprint_cursor_path(
                    prompt,
                    prompt_skipped,
                    indicator_skipped,
                    before_cursor_skipped,
                    use_ansi_coloring,
                    cursor,
                )?;
            } else {
                self.queue_relative_move(cursor, edit_cursor)?;
            }
        }

        Ok(edit_cursor)
    }

    /// Updates prompt origin and offset to handle a screen resize event
    pub(crate) fn handle_resize(&mut self, width: u16, height: u16) {
        self.update_terminal_size((width, height));

        self.invalidate_prompt_start_row();
        self.just_resized = false;

        // `cursor::position()` is blocking and can time out, but a
        // resize happens infrequently enough that we accept the cost.
        // The row stored below is the *cursor* row, not the prompt's
        // screen origin; `just_resized` in `repaint_buffer` re-anchors
        // it on the next paint.
        //
        // Known bug: on iterm2 and kitty, clearing the screen via CMD-K
        // doesn't reset the cursor position — possibly a `position()`
        // bug.
        #[cfg(not(test))]
        {
            // Resize is also the best signal available when a new terminal
            // client attaches to an existing pty. Retry CPR here even after an
            // earlier timeout; success restores precise positioning, while a
            // failure leaves the next repaint in the safe fallback mode.
            let position = self.query_cursor_position(true);
            self.apply_resize_cursor_position(position);
        }
    }

    /// Writes `line` to the terminal followed by `\r\n` and
    /// invalidates the cached prompt anchor since the line scrolls the
    /// terminal independently of the painter.
    pub(crate) fn paint_line(&mut self, line: &str) -> Result<()> {
        // Invalidate up front: a partial write below can still leave
        // bytes in the kernel/tty buffer and displace the cursor.
        self.invalidate_prompt_start_row();
        self.relative_cursor_position = None;
        self.stdout.queue(Print(line))?.queue(Print("\r\n"))?;
        self.stdout.flush()
    }

    /// Clear the screen by printing enough whitespace to start the prompt or
    /// other output back at the first line of the terminal.
    pub(crate) fn clear_screen(&mut self) -> Result<()> {
        self.stdout
            .queue(Clear(ClearType::All))?
            .queue(MoveTo(0, 0))?
            .flush()?;
        self.reset_after_clear(terminal::size()?);
        Ok(())
    }

    pub(crate) fn clear_scrollback(&mut self) -> Result<()> {
        self.stdout
            .queue(Clear(ClearType::All))?
            .queue(Clear(ClearType::Purge))?
            .queue(MoveTo(0, 0))?
            .flush()?;
        self.reset_after_clear(terminal::size()?);
        Ok(())
    }

    /// Park the cursor below the entry on the way out of `read_line`.
    ///
    /// Nothing repaints after this, so whatever sits below the cursor is what
    /// the host prints over. A menu or a hint is not user input, so the erase is
    /// unconditional and only the after-cursor text goes back. Text before the
    /// cursor survives, leaving a rejected command on screen as the record of
    /// what was aborted (#1143).
    pub(crate) fn move_cursor_to_end(&mut self) -> Result<()> {
        let mut cursor = self
            .relative_cursor_position
            .unwrap_or_else(Self::relative_origin);
        if cursor.pending_wrap && self.after_cursor_lines.is_some() {
            self.queue_tracked_print(&mut cursor, "\r\n")?;
        }
        self.stdout.queue(Clear(ClearType::FromCursorDown))?;
        if let Some(after_cursor) = &self.after_cursor_lines {
            let after_cursor = after_cursor.clone();
            self.queue_tracked_print(&mut cursor, &after_cursor)?;
        }
        self.queue_tracked_print(&mut cursor, "\r\n")?;
        self.adopt_cursor_line_as_origin(cursor);
        self.stdout.flush()
    }

    /// Prints an external message
    ///
    /// This function doesn't flush the buffer. So buffer should be flushed
    /// afterwards perhaps by repainting the prompt via `repaint_buffer()`.
    #[cfg(feature = "external_printer")]
    pub(crate) fn print_external_message(
        &mut self,
        messages: Vec<String>,
        _line_buffer: &LineBuffer,
        _prompt: &dyn Prompt,
    ) -> Result<()> {
        let mut cursor = self
            .relative_cursor_position
            .unwrap_or_else(Self::relative_origin);
        self.clear_from_local_row(&mut cursor, 0)?;
        for line in messages {
            let line = coerce_crlf(&line);
            self.queue_tracked_print(&mut cursor, &line)?;
            self.queue_tracked_print(&mut cursor, "\r\n")?;
        }
        self.adopt_cursor_line_as_origin(cursor);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn force_prompt_anchored_for_test(&mut self, row: u16) {
        self.prompt_start_row = PromptStartRow::Verified(row);
    }

    #[cfg(test)]
    pub(crate) fn captured_output_for_test(&self) -> &[u8] {
        self.stdout.captured()
    }

    /// Whether the cached anchor is still trusted, so a test can pin which events cost
    /// a re-verify and which keep #1090's query-free path.
    #[cfg(test)]
    pub(crate) fn prompt_anchor_is_verified_for_test(&self) -> bool {
        matches!(self.prompt_start_row, PromptStartRow::Verified(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menu::MenuEvent;
    use crate::{Color, Completer, Editor, PromptHistorySearch, Suggestion};
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::borrow::Cow;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum MarkerCall {
        PromptPrimary,
        PromptRight,
        CommandInput,
    }

    struct RecordingMarkers {
        calls: Arc<Mutex<Vec<MarkerCall>>>,
    }

    impl SemanticPromptMarkers for RecordingMarkers {
        fn prompt_start(&self, kind: PromptKind) -> Cow<'_, str> {
            let mut calls = self.calls.lock().expect("marker lock poisoned");
            match kind {
                PromptKind::Primary => calls.push(MarkerCall::PromptPrimary),
                PromptKind::Right => calls.push(MarkerCall::PromptRight),
                PromptKind::Secondary => {}
            }
            Cow::Borrowed("")
        }

        fn command_input_start(&self) -> Cow<'_, str> {
            let mut calls = self.calls.lock().expect("marker lock poisoned");
            calls.push(MarkerCall::CommandInput);
            Cow::Borrowed("")
        }
    }

    struct TestPrompt;

    impl Prompt for TestPrompt {
        fn render_prompt_left(&self) -> Cow<'_, str> {
            "> ".into()
        }

        fn render_prompt_right(&self) -> Cow<'_, str> {
            "RP".into()
        }

        fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> Cow<'_, str> {
            "".into()
        }

        fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
            "".into()
        }

        fn render_prompt_history_search_indicator(
            &self,
            _history_search: PromptHistorySearch,
        ) -> Cow<'_, str> {
            "".into()
        }
    }

    #[test]
    fn test_skip_lines() {
        let string = "sentence1\nsentence2\nsentence3\n";

        assert_eq!(skip_buffer_lines(string, 1, None), "sentence2\nsentence3");
        assert_eq!(skip_buffer_lines(string, 2, None), "sentence3");
        assert_eq!(skip_buffer_lines(string, 3, None), "");
        assert_eq!(skip_buffer_lines(string, 4, None), "");
    }

    #[test]
    fn test_skip_lines_no_newline() {
        let string = "sentence1";

        assert_eq!(skip_buffer_lines(string, 0, None), "sentence1");
        assert_eq!(skip_buffer_lines(string, 1, None), "");
    }

    #[test]
    fn test_skip_lines_with_limit() {
        let string = "sentence1\nsentence2\nsentence3\nsentence4\nsentence5";

        assert_eq!(
            skip_buffer_lines(string, 1, Some(1)),
            "sentence2\nsentence3",
        );

        assert_eq!(
            skip_buffer_lines(string, 1, Some(2)),
            "sentence2\nsentence3\nsentence4",
        );

        assert_eq!(
            skip_buffer_lines(string, 2, Some(1)),
            "sentence3\nsentence4",
        );

        assert_eq!(
            skip_buffer_lines(string, 1, Some(10)),
            "sentence2\nsentence3\nsentence4\nsentence5",
        );

        assert_eq!(
            skip_buffer_lines(string, 0, Some(1)),
            "sentence1\nsentence2",
        );

        assert_eq!(skip_buffer_lines(string, 0, Some(0)), "sentence1",);
        assert_eq!(skip_buffer_lines(string, 1, Some(0)), "sentence2",);
    }

    #[test]
    fn test_skip_lines_does_not_leave_half_a_crlf() {
        let string = "line1\r\nline2\r\nline3\r\n";

        assert_eq!(skip_buffer_lines(string, 0, Some(0)), "line1");
        assert_eq!(skip_buffer_lines(string, 1, Some(1)), "line2\r\nline3");
        assert_eq!(skip_buffer_lines(string, 2, None), "line3");
    }

    #[test]
    fn test_select_new_prompt_with_no_state_no_output() {
        assert_eq!(
            select_prompt_row(None, (0, 12)),
            PromptRowSelector::MakeNewPrompt { new_row: 12 }
        );
    }

    #[test]
    fn test_select_new_prompt_with_no_state_but_output() {
        assert_eq!(
            select_prompt_row(None, (3, 12)),
            PromptRowSelector::MakeNewPrompt { new_row: 13 }
        );
    }

    #[test]
    fn test_select_existing_prompt() {
        let state = PainterSuspendedState {
            previous_prompt_rows_range: 11..=13,
            was_flush_at_bottom: false,
        };
        assert_eq!(
            select_prompt_row(Some(&state), (0, 12)),
            PromptRowSelector::UseExistingPrompt { start_row: 11 }
        );
        assert_eq!(
            select_prompt_row(Some(&state), (3, 12)),
            PromptRowSelector::UseExistingPrompt { start_row: 11 }
        );
    }

    #[test]
    fn cursor_position_failure_is_cached_but_a_resize_can_recover() {
        let mut painter = Painter::new(W::sink());

        assert!(
            painter.cursor_position_support.should_query(false),
            "the first read_line must probe CPR"
        );
        assert_eq!(
            painter.record_cursor_position_result(Err(std::io::Error::other("no CPR"))),
            None
        );
        assert_eq!(
            painter.cursor_position_support,
            CursorPositionSupport::Unsupported
        );
        assert!(
            !painter.cursor_position_support.should_query(false),
            "steady-state paints must not repeat the synchronous timeout"
        );
        assert!(
            painter.cursor_position_support.should_query(true),
            "a resize may retry after a terminal client attaches"
        );

        assert_eq!(
            painter.record_cursor_position_result(Ok((4, 7))),
            Some((4, 7))
        );
        assert_eq!(
            painter.cursor_position_support,
            CursorPositionSupport::Supported
        );
    }

    #[test]
    fn initial_cursor_query_failure_adopts_the_current_line_without_output() {
        let mut painter = Painter::new(W::capture());
        painter.update_terminal_size((20, 10));
        let position = painter.record_cursor_position_result(Err(std::io::Error::other("no CPR")));

        painter
            .finish_prompt_position_initialization(None, position)
            .expect("fallback initialization failed");

        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(9));
        let output = String::from_utf8_lossy(painter.stdout.captured());
        assert_eq!(output, "");
        assert_no_frame_crossing_controls(&output, "CPR fallback initialization");
    }

    #[test]
    fn successful_initial_cpr_starts_a_prompt_local_frame() {
        let mut painter = Painter::new(W::capture());
        painter.update_terminal_size((20, 10));
        let position = painter.record_cursor_position_result(Ok((7, 4)));

        painter
            .finish_prompt_position_initialization(None, position)
            .expect("CPR initialization failed");

        // The cursor was part-way through a host-owned line, so preserve it and
        // make the next line the local origin. The measured screen coordinate
        // is metadata only and must never be emitted as a paint destination.
        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(5));
        assert_eq!(
            painter.relative_cursor_position,
            Some(Painter::relative_origin())
        );
        let output = String::from_utf8_lossy(painter.stdout.captured());
        assert_eq!(output, "\r\n");
        assert_no_frame_crossing_controls(&output, "successful CPR initialization");
    }

    #[test]
    fn a_known_clear_position_does_not_reenable_or_query_cpr() {
        let mut painter = Painter::new(W::sink());
        painter.record_cursor_position_result(Err(std::io::Error::other("no CPR")));
        painter.prompt_start_row = PromptStartRow::Stale(8);
        painter.just_resized = true;

        painter.reset_after_clear((20, 10));

        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(0));
        assert_eq!(
            painter.cursor_position_support,
            CursorPositionSupport::Unsupported
        );
        assert!(!painter.just_resized);
    }

    #[test]
    fn a_successful_resize_retry_restores_a_precise_stale_anchor() {
        let mut painter = Painter::new(W::sink());
        painter.record_cursor_position_result(Err(std::io::Error::other("no CPR")));
        painter.prompt_start_row.mark_verified(4);
        painter.invalidate_prompt_start_row();

        let position = painter.record_cursor_position_result(Ok((6, 12)));
        painter.apply_resize_cursor_position(position);

        assert_eq!(
            painter.cursor_position_support,
            CursorPositionSupport::Supported
        );
        assert_eq!(painter.prompt_start_row, PromptStartRow::Stale(12));
        assert!(painter.just_resized);
    }

    #[test]
    fn successful_resize_cpr_is_advisory_during_repaint() {
        let mut painter = Painter::new(W::capture());
        painter.terminal_size = (20, 10);
        painter.prompt_start_row.mark_verified(4);
        painter.relative_cursor_position = Some(Painter::relative_origin());

        let position = painter.record_cursor_position_result(Ok((11, 6)));
        painter.apply_resize_cursor_position(position);
        let lines = make_lines("> ", "", "RP", "one\r\ntwo", "");
        painter
            .repaint_buffer(
                &TestPrompt,
                &lines,
                PromptEditMode::Emacs,
                None,
                false,
                &None,
            )
            .expect("resize repaint failed");

        // Reflow puts the edit cursor one row below the local prompt origin.
        // The CPR row therefore informs the advisory anchor (6 - 1), while all
        // bytes used to repaint remain local.
        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(5));
        assert!(!painter.just_resized);
        let output = String::from_utf8_lossy(painter.stdout.captured());
        assert_no_frame_crossing_controls(&output, "successful resize repaint");
    }

    #[test]
    fn a_stale_anchor_without_cpr_is_rehomed_before_a_full_repaint() {
        let mut painter = Painter::new(W::capture());
        painter.terminal_size = (20, 10);
        painter.cursor_position_support = CursorPositionSupport::Unsupported;
        painter.prompt_start_row = PromptStartRow::Stale(3);
        let lines = make_lines("> ", "", "RP", "one\r\ntwo\r\nthree", "");

        painter
            .repaint_buffer(
                &TestPrompt,
                &lines,
                PromptEditMode::Emacs,
                None,
                false,
                &None,
            )
            .expect("fallback repaint failed");

        let required_lines = painter.last_required_lines;
        assert!(required_lines > 1);
        assert_eq!(
            painter.prompt_start_row,
            PromptStartRow::Verified(10 - required_lines)
        );
        let output = String::from_utf8_lossy(painter.stdout.captured());
        let fallback_newline = output.find("\r\n").expect("fallback CRLF missing");
        let prompt = output.find("> ").expect("prompt missing");
        assert!(
            fallback_newline < prompt,
            "fallback must establish a fresh local origin before painting; emitted {output:?}"
        );
        assert_no_frame_crossing_controls(&output, "stale CPR fallback repaint");
    }

    #[cfg(feature = "external_printer")]
    #[test]
    fn external_print_without_cpr_rehomes_on_the_following_repaint() {
        let mut painter = Painter::new(W::capture());
        painter.terminal_size = (20, 10);
        painter.cursor_position_support = CursorPositionSupport::Unsupported;
        painter.prompt_start_row.mark_verified(5);

        painter
            .print_external_message(
                vec!["background notice".to_owned()],
                &LineBuffer::default(),
                &TestPrompt,
            )
            .expect("external print failed");
        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(6));

        let lines = make_lines("> ", "", "", "", "");
        painter
            .repaint_buffer(
                &TestPrompt,
                &lines,
                PromptEditMode::Emacs,
                None,
                false,
                &None,
            )
            .expect("fallback repaint failed");

        assert_eq!(painter.prompt_start_row, PromptStartRow::Verified(6));
        let output = String::from_utf8_lossy(painter.stdout.captured());
        assert!(output.contains("background notice"));
        assert_no_frame_crossing_controls(&output, "external printer and following repaint");

        const PREFIX: &str = "auth-1 | ";
        let stream = format!("{PREFIX}boot\r\n{}", add_compose_prefix(&output, PREFIX));
        let replayed = replay(&stream, 20 + PREFIX.len() as u16, false);
        assert_eq!(
            replayed.rows.first().map(String::as_str),
            Some("auth-1 | boot"),
            "external print changed the preceding host log row"
        );
        assert!(
            replayed
                .rows
                .iter()
                .skip(1)
                .all(|row| row.starts_with(PREFIX)),
            "external print erased a Compose prefix: {:?}",
            replayed.rows
        );
    }

    // Regression test for nushell/reedline#1130.
    //
    // A multi-line prompt flush against the bottom of the screen is suspended
    // for an fzf-style keybinding. The program scrolls the terminal and returns
    // with the cursor pinned on the bottom row, still inside the stored range.
    // Re-using the old anchor would redraw the prompt over the scrolled-up
    // output, so the ambiguous bottom case must make a fresh prompt instead.
    #[test]
    fn test_select_prompt_row_does_not_reuse_when_flush_at_bottom() {
        let state = PainterSuspendedState {
            previous_prompt_rows_range: 5..=7,
            was_flush_at_bottom: true,
        };
        assert_eq!(
            select_prompt_row(Some(&state), (0, 7)),
            PromptRowSelector::MakeNewPrompt { new_row: 7 }
        );
    }

    // The flush-at-bottom fact is captured against the screen the prompt was
    // suspended on, since whatever runs in between may resize the terminal.
    #[rstest]
    #[case::well_above_bottom(2, 3, false)]
    #[case::reaches_last_row(5, 2, true)]
    // A prompt at the very bottom pushes `final_row` past the last visible row.
    #[case::overshoots_last_row(5, 3, true)]
    fn test_state_before_suspension_records_flush_at_bottom(
        #[case] start_row: u16,
        #[case] required_lines: u16,
        #[case] expected: bool,
    ) {
        let mut painter = Painter::new(W::sink());
        painter.handle_resize(80, 8); // rows 0..=7
        painter.prompt_start_row.mark_verified(start_row);
        painter.last_required_lines = required_lines;

        assert_eq!(
            painter.state_before_suspension().was_flush_at_bottom,
            expected
        );
    }

    fn base_snapshot() -> RenderSnapshot {
        RenderSnapshot {
            screen_width: 20,
            screen_height: 10,
            prompt_start_row: 0,
            prompt_height: 1,
            large_buffer: false,
            prompt_str_left: "> ".to_string(),
            prompt_indicator: "".to_string(),
            before_cursor: "".to_string(),
            after_cursor: "".to_string(),
            first_buffer_col: 2,
            menu_active: false,
            menu_start_row: None,
            large_buffer_extra_rows_after_prompt: None,
            large_buffer_offset: None,
            right_prompt: None,
        }
    }

    #[test]
    fn test_screen_to_buffer_simple() {
        let mut snapshot = base_snapshot();
        snapshot.before_cursor = "hello world".to_string();

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 2, 0), Some(0));
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 3, 0), Some(1));
    }

    #[test]
    fn test_clicks_past_eol_clamps() {
        let mut snapshot = base_snapshot();
        snapshot.before_cursor = "hi".to_string();

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 10, 0), Some(2));
    }

    #[test]
    fn test_wrapped_line_mapping() {
        let mut snapshot = base_snapshot();
        snapshot.screen_width = 5;
        snapshot.before_cursor = "abcdef".to_string();

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 1, 1), Some(4));
    }

    #[test]
    fn test_multiline_mapping() {
        let mut snapshot = base_snapshot();
        snapshot.before_cursor = "ab\ncd".to_string();

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 1, 1), Some(4));
    }

    #[test]
    fn test_large_buffer_skips_lines() {
        let mut snapshot = base_snapshot();
        snapshot.large_buffer = true;
        snapshot.first_buffer_col = 0;
        snapshot.before_cursor = "line1\nline2\nline3".to_string();
        snapshot.large_buffer_extra_rows_after_prompt = Some(1);

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 0, 0), Some(6));
    }

    #[test]
    fn test_click_in_right_prompt_ignored() {
        let mut snapshot = base_snapshot();
        snapshot.before_cursor = "hello".to_string();
        snapshot.right_prompt = Some(RightPromptBounds {
            row: 0,
            start_col: 10,
            end_col: 12,
        });

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 10, 0), None);
    }

    #[test]
    fn test_click_in_menu_ignored() {
        let mut snapshot = base_snapshot();
        snapshot.menu_active = true;
        snapshot.menu_start_row = Some(2);

        let painter = Painter::new(W::sink());
        assert_eq!(painter.screen_to_buffer_offset(&snapshot, 0, 2), None);
    }

    fn make_painter(width: u16, height: u16, large_buffer: bool) -> Painter {
        let mut p = Painter::new(W::sink());
        p.terminal_size = (width, height);
        p.prompt_start_row.mark_verified(0);
        p.prompt_height = 1;
        p.large_buffer = large_buffer;
        p
    }

    fn make_lines<'a>(
        left: &'a str,
        indicator: &'a str,
        right: &'a str,
        before: &'a str,
        after: &'a str,
    ) -> PromptLines<'a> {
        PromptLines {
            prompt_str_left: Cow::Borrowed(left),
            prompt_str_right: Cow::Borrowed(right),
            prompt_indicator: Cow::Borrowed(indicator),
            before_cursor: Cow::Borrowed(before),
            after_cursor: Cow::Borrowed(after),
            hint: Cow::Borrowed(""),
            right_prompt_on_last_line: false,
        }
    }

    /// Ordinary rendering owns only its prompt-local frame. Commands that set
    /// a screen row/column, restore an earlier cursor, or scroll the whole
    /// terminal can cross a line prefix injected by Docker Compose and rewrite
    /// historical log rows, so none may appear in a captured paint stream.
    fn assert_no_frame_crossing_controls(output: &str, context: &str) {
        let bytes = output.as_bytes();
        let mut forbidden = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i] == b'\r' && bytes.get(i + 1) != Some(&b'\n') {
                forbidden.push("bare CR".to_string());
                i += 1;
                continue;
            }

            if bytes[i] != b'\x1b' {
                i += 1;
                continue;
            }

            if let Some(next) = bytes.get(i + 1).copied() {
                if matches!(next, b'7' | b'8' | b'D' | b'E' | b'M') {
                    forbidden.push(format!("ESC {}", char::from(next)));
                    i += 2;
                    continue;
                }
            }

            if bytes.get(i + 1) != Some(&b'[') {
                i += 1;
                continue;
            }

            let mut end = i + 2;
            while end < bytes.len() && !(0x40..=0x7e).contains(&bytes[end]) {
                end += 1;
            }
            if end == bytes.len() {
                break;
            }

            // CUP/HVP, absolute row/column controls, next/previous-line,
            // save/restore, and whole-screen scroll controls.
            if matches!(
                bytes[end],
                b'H' | b'f' | b'G' | b'd' | b'`' | b'E' | b'F' | b's' | b'u' | b'S' | b'T'
            ) {
                forbidden.push(
                    String::from_utf8_lossy(&bytes[i..=end])
                        .escape_debug()
                        .to_string(),
                );
            }
            i = end + 1;
        }

        assert!(
            forbidden.is_empty(),
            "{context} emitted frame-crossing terminal controls {forbidden:?}: {output:?}"
        );
    }

    /// Paint once into a capture buffer, returning the bytes emitted, the rows
    /// the paint reserved, and whether it took the large-buffer path.
    ///
    /// `anchor_row` is marked verified so the painter takes the no-drift path
    /// and never queries the real terminal. `large_buffer` comes back so a case
    /// can assert it exercised the path it meant to, since it turns on at
    /// `required_lines >= screen_height` rather than at anything stated here.
    fn capture_repaint(lines: &PromptLines, anchor_row: u16) -> (String, u16, bool) {
        let mut p = Painter::new(W::capture());
        p.terminal_size = (20, 10);
        p.prompt_start_row.mark_verified(anchor_row);
        p.relative_cursor_position = Some(Painter::relative_origin());
        p.prompt_height = 1;
        p.repaint_buffer(
            &TestPrompt,
            lines,
            PromptEditMode::Default,
            None,
            false,
            &None,
        )
        .expect("repaint_buffer failed");
        (
            String::from_utf8_lossy(p.stdout.captured()).into_owned(),
            p.last_required_lines,
            p.large_buffer,
        )
    }

    /// What a terminal ends up in after a byte stream.
    #[derive(Debug)]
    struct Replayed {
        /// Row, column, and whether a wrap is deferred. Compared as a unit,
        /// since a cursor that agrees on two of the three is still ambiguous.
        cursor: (u16, u16, bool),
        /// The highest row a glyph reached, to check against the rows reserved.
        max_written: u16,
        /// Rows right-trimmed and concatenated with no separator, so a case can
        /// state the text it expects without also stating where it wrapped.
        screen: String,
        /// Individual right-trimmed rows, for assertions about host-owned log
        /// lines and prefixes that must survive a repaint.
        rows: Vec<String>,
    }

    /// Replay `bytes` into a character grid.
    ///
    /// `save_carries_pending` picks which way DECSC/DECRC treat a deferred wrap,
    /// so the same stream can be read as either kind of terminal would.
    fn replay(bytes: &str, width: u16, save_carries_pending: bool) -> Replayed {
        fn put(r: u16, c: u16, ch: char, width: u16, g: &mut Vec<Vec<char>>) {
            while g.len() <= r as usize {
                g.push(vec![' '; width as usize]);
            }
            g[r as usize][c as usize] = ch;
        }

        let (mut row, mut col, mut pending) = (0u16, 0u16, false);
        let (mut srow, mut scol, mut spend) = (0u16, 0u16, false);
        let mut max_written = 0u16;
        let mut grid: Vec<Vec<char>> = vec![];
        let b: Vec<char> = bytes.chars().collect();
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                '\x1b' if i + 1 < b.len() && b[i + 1] == '7' => {
                    (srow, scol, spend) = (row, col, pending && save_carries_pending);
                    i += 2;
                }
                '\x1b' if i + 1 < b.len() && b[i + 1] == '8' => {
                    (row, col, pending) = (srow, scol, spend);
                    i += 2;
                }
                '\x1b' if i + 1 < b.len() && b[i + 1] == '[' => {
                    let mut j = i + 2;
                    while j < b.len() && !('\x40'..='\x7e').contains(&b[j]) {
                        j += 1;
                    }
                    let params: String = b[i + 2..j.min(b.len())].iter().collect();
                    if j < b.len() && b[j] == 'H' {
                        let mut it = params.split(';');
                        row = it
                            .next()
                            .unwrap_or("1")
                            .parse::<u16>()
                            .unwrap_or(1)
                            .saturating_sub(1);
                        col = it
                            .next()
                            .unwrap_or("1")
                            .parse::<u16>()
                            .unwrap_or(1)
                            .saturating_sub(1);
                        pending = false;
                    } else if j < b.len() && b[j] == 'G' {
                        col = params.parse::<u16>().unwrap_or(1).saturating_sub(1);
                        pending = false;
                    } else if j < b.len() && matches!(b[j], 'A' | 'B' | 'C' | 'D') {
                        let amount = params.parse::<u16>().unwrap_or(1);
                        match b[j] {
                            'A' => row = row.saturating_sub(amount),
                            'B' => row = row.saturating_add(amount),
                            'C' => col = col.saturating_add(amount).min(width.saturating_sub(1)),
                            'D' => col = col.saturating_sub(amount),
                            _ => unreachable!(),
                        }
                        pending = false;
                    } else if j < b.len() && b[j] == 'J' && matches!(params.as_str(), "" | "0") {
                        // Erase-below (`ED(0)`): rest of this row, then every row
                        // under it. Cells only, so cursor and `max_written` stay
                        // put; `screen` is the one field an erase can move.
                        if let Some(line) = grid.get_mut(row as usize) {
                            for cell in line.iter_mut().skip(col as usize) {
                                *cell = ' ';
                            }
                        }
                        grid.truncate(row as usize + 1);
                    }
                    i = j + 1;
                }
                '\n' => {
                    row += 1;
                    col = 0;
                    pending = false;
                    i += 1;
                }
                '\r' => {
                    col = 0;
                    pending = false;
                    i += 1;
                }
                ch => {
                    if pending {
                        row += 1;
                        col = 0;
                        pending = false;
                    }
                    put(row, col, ch, width, &mut grid);
                    max_written = max_written.max(row);
                    if col + 1 >= width {
                        pending = true;
                    } else {
                        col += 1;
                    }
                    i += 1;
                }
            }
        }
        let rows: Vec<String> = grid
            .iter()
            .map(|r| r.iter().collect::<String>().trim_end().to_string())
            .collect();
        let screen = rows.concat();
        Replayed {
            cursor: (row, col, pending),
            max_written,
            screen,
            rows,
        }
    }

    /// Model Compose's line-oriented log prefixing around a container's raw
    /// terminal stream. The prefix is outside reedline's local coordinate
    /// system but occupies real cells when the combined stream is replayed.
    fn add_compose_prefix(output: &str, prefix: &str) -> String {
        format!(
            "{prefix}{}",
            output.replace("\r\n", &format!("\r\n{prefix}"))
        )
    }

    #[test]
    fn ordinary_paint_branches_never_cross_the_prompt_local_frame() {
        let small = make_lines("> ", "", "RP", "hello", " tail");
        let (small_output, _, small_is_large) = capture_repaint(&small, 4);
        assert!(!small_is_large);
        assert_no_frame_crossing_controls(&small_output, "small-buffer repaint");

        let after = "tail\r\n".repeat(30);
        let large = make_lines("> ", "", "RP", "hello", &after);
        let (large_output, _, large_is_large) = capture_repaint(&large, 4);
        assert!(large_is_large);
        assert_no_frame_crossing_controls(&large_output, "large-buffer repaint");

        let multiline_prompt = format!("ab\n{}", "x".repeat(20));
        let right_prompt = make_lines(&multiline_prompt, "", "R", "Z", "after");
        let (right_output, _, _) = capture_repaint(&right_prompt, 4);
        assert_no_frame_crossing_controls(&right_output, "right-prompt repaint");
    }

    #[test]
    fn compose_prefixes_and_previous_log_rows_survive_relative_repaint() {
        const PREFIX: &str = "auth-1 | ";
        // `PromptLines::new` normalizes editable buffer newlines to CRLF before
        // they reach the painter; use that production representation here.
        let lines = make_lines("> ", "", "RP", "one", "\r\ntail");
        let (output, ..) = capture_repaint(&lines, 1);
        assert_no_frame_crossing_controls(&output, "Compose-prefixed repaint");

        let stream = format!("{PREFIX}boot\r\n{}", add_compose_prefix(&output, PREFIX));
        let replayed = replay(&stream, 20 + PREFIX.len() as u16, false);

        assert_eq!(
            replayed.rows.first().map(String::as_str),
            Some("auth-1 | boot"),
            "repaint changed the preceding host log row; emitted {output:?}"
        );
        assert!(
            replayed
                .rows
                .get(1)
                .is_some_and(|row| row.starts_with(PREFIX)),
            "repaint erased the prompt row's Compose prefix: {:?}",
            replayed.rows
        );
        assert!(
            replayed
                .rows
                .get(2)
                .is_some_and(|row| row.starts_with(PREFIX)),
            "repaint erased a continuation row's Compose prefix: {:?}",
            replayed.rows
        );
    }

    /// The three assertions below must hold *together*: two earlier fixes for
    /// this bug each satisfied some and broke another, so any one of them alone
    /// passes a broken painter.
    ///
    /// `"> "` is 2 columns of a 20-column terminal, so `n == 18` and `n == 38`
    /// land the cursor exactly on a margin.
    #[rstest]
    #[case(17, "")]
    #[case(18, "")]
    #[case(19, "")]
    #[case(37, "")]
    #[case(38, "")]
    #[case(17, "XYZ")]
    #[case(18, "XYZ")]
    #[case(19, "XYZ")]
    #[case(38, "XYZ")]
    fn a_paint_pins_the_cursor_without_disturbing_the_screen(
        #[case] n: usize,
        #[case] after: &str,
    ) {
        let before = "a".repeat(n);
        let lines = make_lines("> ", "", "", &before, after);
        let (out, reserved, _) = capture_repaint(&lines, 0);

        let (first, second) = (replay(&out, 20, true), replay(&out, 20, false));

        assert_eq!(
            first.cursor, second.cursor,
            "n={n} after={after:?}: cursor depends on how DECSC treats the \
             pending-wrap flag; emitted {out:?}"
        );
        let written = first.max_written.max(second.max_written);
        assert!(
            written < reserved,
            "n={n} after={after:?}: wrote to row {written} with only {reserved} \
             row(s) reserved; the next erase-below would leave it on screen"
        );
        let expected = format!("> {before}{after}");
        assert_eq!(
            first.screen, expected,
            "n={n} after={after:?}: screen was corrupted"
        );
        assert_eq!(
            second.screen, expected,
            "n={n} after={after:?}: screen was corrupted"
        );
    }

    /// The large-buffer path prints line-skipped text, so the margin has to be
    /// judged from what was emitted rather than from the whole buffer.
    ///
    /// The bulk sits *after* the cursor deliberately: on this 20x10 terminal the
    /// ~200 columns that turn `large_buffer` on would, placed before the cursor,
    /// also fill the screen and put the margin out of reach (see
    /// [`a_paint_that_fills_the_screen_does_not_move_off_it`]). `n == 37` is the
    /// off-margin control.
    #[rstest]
    #[case(38)]
    #[case(58)]
    #[case(78)]
    #[case(37)]
    fn a_large_buffer_paint_pins_the_cursor_too(#[case] n: usize) {
        let before = "a".repeat(n);
        let after = "y".repeat(220);
        let lines = make_lines("> ", "", "", &before, &after);
        let (out, _reserved, large) = capture_repaint(&lines, 0);
        assert!(large, "n={n}: meant to exercise the large-buffer path");

        let (first, second) = (replay(&out, 20, true), replay(&out, 20, false));
        assert_eq!(
            first.cursor, second.cursor,
            "n={n}: cursor depends on how DECSC treats the pending-wrap flag; \
             emitted {out:?}"
        );
    }

    /// Regression: `skip_buffer_lines` trimmed the prompt's trailing newline, so
    /// a two-line prompt collapsed onto one row and the input painted over it.
    ///
    /// Content after the cursor stands in for the tall completion menu that
    /// reaches this in practice: both turn `large_buffer` on without adding rows
    /// before the cursor, so `extra_rows` stays 0 and the whole prompt is still
    /// meant to be drawn.
    #[test]
    fn a_large_buffer_keeps_the_prompts_trailing_newline() {
        let after = "y".repeat(200);
        let lines = make_lines("ab\n", "> ", "", "Z", &after);
        let (out, _reserved, large) = capture_repaint(&lines, 0);
        assert!(large, "meant to exercise the large-buffer path");

        assert!(
            out.contains("ab\r\n> Z"),
            "indicator and input landed on the prompt's row; emitted {out:?}"
        );
    }

    /// The right prompt makes a relative round trip away from the current edit
    /// cursor. If that cursor has a deferred wrap, moving back can reach the
    /// same cell but cannot recreate the terminal's pending-wrap flag, so the
    /// path from the local origin has to be repainted.
    ///
    /// Reaching it needs a *multi-line* left prompt. Placement is judged by
    /// `estimate_right_prompt_line_width`, which for a one-row prompt adds the
    /// indicator and input width and so overflows the margin case out of
    /// existence; past one row it counts only the first line. A short first line
    /// therefore leaves room for the right prompt while the last line still
    /// fills the width, which is exactly when the return position is pending.
    #[test]
    fn a_right_prompt_paint_pins_the_cursor_too() {
        // 20 columns: a 2-column first line, a second that fills the width.
        let left = format!("ab\n{}", "x".repeat(20));
        let lines = make_lines(&left, "", "R", "Z", "");
        let (out, reserved, large) = capture_repaint(&lines, 0);
        assert!(!large, "meant to exercise the small-buffer path");

        let (first, second) = (replay(&out, 20, true), replay(&out, 20, false));

        assert_eq!(
            first.cursor, second.cursor,
            "cursor depends on how a saved pending-wrap flag is interpreted; \
             emitted {out:?}"
        );
        let written = first.max_written.max(second.max_written);
        assert!(
            written < reserved,
            "wrote to row {written} with only {reserved} row(s) reserved"
        );
        // Row 0 is the prompt's first line with `R` in the last cell, row 1 the
        // filled line, row 2 the input. `replay` concatenates right-trimmed rows.
        let expected = format!("ab{}R{}Z", " ".repeat(17), "x".repeat(20));
        assert_eq!(first.screen, expected, "screen was corrupted");
        assert_eq!(second.screen, expected, "screen was corrupted");
    }

    /// The same relative round trip on the large-buffer path, which prints a
    /// *skipped* prompt and has to judge the margin from that rather than from
    /// `lines`.
    ///
    /// The two flags look mutually exclusive and are not: `large_buffer` is
    /// `required_lines >= screen_height` over the whole content, while
    /// `extra_rows` counts only what precedes the cursor. Content after the
    /// cursor therefore turns the large-buffer path on while leaving the prompt
    /// unscrolled, which is the one combination that still paints a right
    /// prompt (`extra_rows > 0` suppresses it).
    #[test]
    fn a_large_buffer_right_prompt_paint_pins_the_cursor_too() {
        let left = format!("ab\n{}", "x".repeat(20));
        // 20x10 terminal: the trailing content pushes `required_lines` past the
        // height without adding rows before the cursor.
        let after = "y".repeat(200);
        let lines = make_lines(&left, "", "R", "Z", &after);
        let (out, _reserved, large) = capture_repaint(&lines, 0);
        assert!(large, "meant to exercise the large-buffer path");

        let (first, second) = (replay(&out, 20, true), replay(&out, 20, false));

        assert_eq!(
            first.cursor, second.cursor,
            "cursor depends on how a saved pending-wrap flag is interpreted; \
             emitted {out:?}"
        );
        // Asserted against each other rather than a literal: what a large buffer
        // puts on screen is a trimmed window, but it must not differ by terminal.
        assert_eq!(
            first.screen, second.screen,
            "screen depends on the DECSC convention"
        );
    }

    /// Content that fills the screen exactly used to produce an absolute target
    /// on a row the terminal had not scrolled into existence. A local renderer
    /// must not encode that destination at all.
    ///
    /// 20x10 terminal, `"> "` is 2 columns, so `n == 198` fills all ten rows.
    #[rstest]
    #[case(198)]
    #[case(218)]
    #[case(238)]
    fn a_paint_that_fills_the_screen_does_not_move_off_it(#[case] n: usize) {
        let before = "a".repeat(n);
        let lines = make_lines("> ", "", "", &before, "");
        let (out, ..) = capture_repaint(&lines, 0);

        assert_no_frame_crossing_controls(&out, &format!("screen-filling repaint, n={n}"));
    }

    #[test]
    fn repaint_at_row_0_does_not_erase_from_home_cell() {
        // tmux `scroll-on-clear` (default on) snapshots the whole screen when
        // erase-below starts at the home cell. Guard column 1, erase, then move
        // back locally; the first prompt cell immediately repaints the guard.
        let (out, ..) = capture_repaint(&make_lines("> ", "", "RP", "hello", ""), 0);
        assert!(
            out.contains("\x1b[1C\x1b[J\x1b[1D"),
            "expected a guarded local erase, which avoids tmux snapshotting the prompt; emitted: {out:?}"
        );
        assert_no_frame_crossing_controls(&out, "repaint at local row zero");
    }

    #[test]
    fn repaint_from_a_nonzero_screen_row_clears_only_the_local_frame() {
        let lines = make_lines("> ", "", "RP", "hello", "");
        let (at_home, ..) = capture_repaint(&lines, 0);
        let (out, ..) = capture_repaint(&lines, 3);

        assert!(
            out.contains("\x1b[J"),
            "expected the local frame to be erased before repaint; emitted: {out:?}"
        );
        assert_eq!(out, at_home, "absolute anchor row leaked into paint bytes");
        assert_no_frame_crossing_controls(&out, "repaint from nonzero screen row");
    }

    /// Minimal `Menu` reporting a fixed block of rows. `menu_string` draws them,
    /// `menu_required_lines` books them, and both derive from the same string so
    /// a test cannot describe a menu that draws more rows than it reserved.
    /// Everything else is unreachable in these tests.
    struct TestMenu(String);

    impl Menu for TestMenu {
        fn menu_string(&self, _available_lines: u16, _use_ansi_coloring: bool) -> String {
            self.0.clone()
        }
        fn is_active(&self) -> bool {
            true
        }
        fn set_active(&mut self, _active: bool) {}
        fn clear_input(&mut self) {}
        fn menu_event(&mut self, _event: MenuEvent) {
            unimplemented!()
        }
        fn can_quick_complete(&self) -> bool {
            unimplemented!()
        }
        fn can_partially_complete(
            &mut self,
            _values_updated: bool,
            _editor: &mut Editor,
            _completer: &mut dyn Completer,
        ) -> bool {
            unimplemented!()
        }
        fn update_values(&mut self, _editor: &mut Editor, _completer: &mut dyn Completer) {
            unimplemented!()
        }
        fn reset_position(&mut self) {
            unimplemented!()
        }
        fn update_working_details(
            &mut self,
            _editor: &mut Editor,
            _completer: &mut dyn Completer,
            _painter: &Painter,
        ) {
            unimplemented!()
        }
        fn replace_in_buffer(&self, _editor: &mut Editor) {
            unimplemented!()
        }
        fn menu_required_lines(&self, _terminal_columns: u16) -> u16 {
            self.0.lines().count() as u16
        }
        fn min_rows(&self) -> u16 {
            unimplemented!()
        }
        fn get_values(&self) -> &[Suggestion] {
            unimplemented!()
        }
    }

    /// Paint a menu into a capture buffer and return the emitted bytes, with the
    /// menu starting at `menu_start_row` (None exercises the `unwrap_or(0)`).
    fn capture_print_menu(menu: &dyn Menu, menu_start_row: Option<u16>) -> String {
        let mut p = Painter::new(W::capture());
        p.terminal_size = (20, 10);
        let layout = PromptLayout {
            extra_rows: 0,
            extra_rows_after_prompt: 0,
            large_buffer_offset: None,
            right_prompt: None,
            menu_start_row,
            first_buffer_col: 0,
        };
        let mut cursor = Painter::relative_origin();
        p.print_menu(menu, false, &layout, &mut cursor)
            .expect("print_menu failed");
        String::from_utf8_lossy(p.stdout.captured()).into_owned()
    }

    /// Paint, then leave through the exit path, replaying the whole stream.
    /// Same setup as `capture_repaint`, so the two stay comparable.
    fn capture_repaint_then_exit(lines: &PromptLines, menu: Option<&ReedlineMenu>) -> Replayed {
        let mut p = Painter::new(W::capture());
        p.terminal_size = (20, 10);
        p.prompt_start_row.mark_verified(0);
        p.relative_cursor_position = Some(Painter::relative_origin());
        p.prompt_height = 1;
        p.repaint_buffer(
            &TestPrompt,
            lines,
            PromptEditMode::Default,
            menu,
            false,
            &None,
        )
        .expect("repaint_buffer failed");
        p.move_cursor_to_end().expect("move_cursor_to_end failed");

        replay(&String::from_utf8_lossy(p.stdout.captured()), 20, false)
    }

    /// Regression test for nushell/reedline#1143.
    ///
    /// The exit path is the last thing to touch the screen, so what it leaves
    /// below the cursor is what the host prints over. A menu and a hint both sit
    /// there and neither is user input, so both go; the command stays, as the
    /// record of what was rejected. One bug, two symptoms: `print_small_buffer`
    /// draws menu or hint from the same branch, and the erase used to be gated
    /// on after-cursor text, which is unrelated to either.
    ///
    /// Every case types `pwd`, so the surviving screen is always `"> pwd"`.
    #[rstest]
    #[case::menu_is_erased(Some("item1\nitem2\nitem3"), "pwd", "", "")]
    #[case::hint_is_erased(None, "pwd", "", " --help")]
    // Mid-buffer the erase takes the trailing "d" with it, so this also pins
    // that the after-cursor text is put back afterwards.
    #[case::after_cursor_text_survives(Some("item1\nitem2\nitem3"), "pw", "d", "")]
    #[case::nothing_below_the_cursor_at_all(None, "pwd", "", "")]
    fn exit_erases_below_the_cursor_but_keeps_the_command(
        #[case] menu_rows: Option<&str>,
        #[case] before: &str,
        #[case] after: &str,
        #[case] hint: &str,
    ) {
        let menu = menu_rows
            .map(|rows| ReedlineMenu::EngineCompleter(Box::new(TestMenu(rows.to_string()))));
        let mut lines = make_lines("> ", "", "", before, after);
        lines.hint = Cow::Borrowed(hint);

        let replayed = capture_repaint_then_exit(&lines, menu.as_ref());

        assert_eq!(
            replayed.screen, "> pwd",
            "exit left content below the cursor for the host to print over, or \
             dropped the command it should have kept"
        );
        // Prompt and command occupy row 0 alone once everything below is gone.
        assert_eq!(
            replayed.cursor.0, 1,
            "exit parked the cursor on row {}, not on the first free row",
            replayed.cursor.0
        );
    }

    #[test]
    fn print_menu_at_row_0_does_not_erase_from_home_cell() {
        // Same tmux trigger as the prompt path: a menu drawn at local row 0
        // must use the guarded erase sequence (#1062).
        let menu = TestMenu("item1\nitem2".to_string());
        let out = capture_print_menu(&menu, Some(0));
        assert!(
            out.contains("\x1b[1C\x1b[J\x1b[1D"),
            "expected a guarded local menu erase; emitted: {out:?}"
        );
        assert_no_frame_crossing_controls(&out, "menu repaint at local row zero");
    }

    #[test]
    fn print_menu_none_start_row_treated_as_row_0() {
        // `unwrap_or(0)` makes a None start row clear from local row 0, so it
        // must honour the same guard.
        let menu = TestMenu("item1".to_string());
        let out = capture_print_menu(&menu, None);
        assert!(
            out.contains("\x1b[1C\x1b[J\x1b[1D"),
            "expected a guarded local menu erase; emitted: {out:?}"
        );
        assert_no_frame_crossing_controls(&out, "menu repaint with implicit local row zero");
    }

    #[test]
    fn test_layout_small_buffer_defaults() {
        let painter = make_painter(20, 10, false);
        let lines = make_lines("> ", "", "", "hello", "");
        let layout = painter.compute_layout(&lines, None);

        assert_eq!(layout.extra_rows, 0);
        assert_eq!(layout.extra_rows_after_prompt, 0);
        assert_eq!(layout.large_buffer_offset, None);
        assert_eq!(layout.first_buffer_col, 2); // "> " is 2 chars wide
        assert_eq!(layout.menu_start_row, None);
    }

    #[test]
    fn test_layout_right_prompt_rendered() {
        let painter = make_painter(40, 10, false);
        let lines = make_lines("> ", "", "RP", "hi", "");
        let layout = painter.compute_layout(&lines, None);

        let rp = layout
            .right_prompt
            .expect("right prompt should be rendered");
        assert_eq!(rp.row, 0);
        assert_eq!(rp.start_col, 38); // 40 - 2
        assert_eq!(rp.end_col, 40);
    }

    #[test]
    fn test_layout_right_prompt_hidden_when_input_too_wide() {
        let painter = make_painter(10, 10, false);
        // Prompt "> " (2) + before "12345678" (8) = 10 which equals start_position (10-2=8)
        // input_width(10) > start_position(8) so right prompt should not render
        let lines = make_lines("> ", "", "RP", "12345678", "");
        let layout = painter.compute_layout(&lines, None);

        assert!(layout.right_prompt.is_none());
    }

    #[test]
    fn test_layout_large_buffer_extra_rows() {
        // Screen is 5 lines tall, buffer content exceeds it.
        // prompt_lines_with_wrap(""> ") = 0
        // prompt_indicator_lines("") = 0
        // before_cursor has 7 lines
        // total_lines_before = 0 + 0 + 7 - 1 = 6
        // extra_rows = 6 - 5 = 1
        // extra_rows_after_prompt = 1 - 0 = 1
        let painter = make_painter(20, 5, true);
        let lines = make_lines("> ", "", "", "l1\nl2\nl3\nl4\nl5\nl6\nl7", "");
        let layout = painter.compute_layout(&lines, None);

        assert_eq!(layout.extra_rows, 1);
        assert_eq!(layout.extra_rows_after_prompt, 1);
        assert_eq!(layout.first_buffer_col, 0); // scrolled, so col 0
    }

    #[test]
    fn test_layout_right_prompt_suppressed_in_large_buffer() {
        // When extra_rows > 0 the prompt has scrolled off, so right prompt
        // should not be rendered — this was a bug in the old render_snapshot.
        let painter = make_painter(20, 5, true);
        let lines = make_lines("> ", "", "RP", "l1\nl2\nl3\nl4\nl5\nl6\nl7", "");
        let layout = painter.compute_layout(&lines, None);

        assert!(layout.extra_rows > 0);
        assert!(layout.right_prompt.is_none());
    }

    #[test]
    fn test_layout_large_buffer_no_scroll_keeps_right_prompt() {
        // Large buffer flag set but content fits — extra_rows == 0
        // Right prompt should still render
        let painter = make_painter(20, 10, true);
        let lines = make_lines("> ", "", "RP", "short", "");
        let layout = painter.compute_layout(&lines, None);

        assert_eq!(layout.extra_rows, 0);
        assert!(layout.right_prompt.is_some());
    }

    #[test]
    fn test_layout_first_buffer_col_with_multiline_prompt() {
        let painter = make_painter(20, 10, false);
        // Multi-line prompt: last line is "$ " (2 chars)
        let lines = make_lines("line1\n$ ", "", "", "hello", "");
        let layout = painter.compute_layout(&lines, None);

        assert_eq!(layout.first_buffer_col, 2);
    }

    #[test]
    fn test_prompt_marker_order_in_small_buffer() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let markers = RecordingMarkers {
            calls: Arc::clone(&calls),
        };

        let mut painter = Painter::new(W::sink());
        painter.terminal_size = (20, 10);
        painter.prompt_start_row.mark_verified(0);
        painter.prompt_height = 1;
        painter.set_semantic_markers(Some(Box::new(markers)));

        let prompt = TestPrompt;
        let lines = PromptLines::new(&prompt, PromptEditMode::Default, None, "", "", "");
        let layout = painter.compute_layout(&lines, None);

        let mut cursor = Painter::relative_origin();
        painter
            .print_small_buffer(&prompt, &lines, None, false, &layout, &mut cursor)
            .expect("print_small_buffer failed");

        let recorded = calls.lock().expect("marker lock poisoned").clone();
        assert_eq!(
            recorded,
            vec![
                MarkerCall::PromptPrimary,
                MarkerCall::PromptRight,
                MarkerCall::CommandInput
            ]
        );
    }

    const SGR_GREEN: &str = "\x1b[92m"; // DEFAULT_PROMPT_COLOR (LightGreen, palette 10)
    const SGR_CYAN: &str = "\x1b[96m"; // DEFAULT_INDICATOR_COLOR (LightCyan, palette 14)
    const SGR_PURPLE: &str = "\x1b[35m"; // DEFAULT_PROMPT_RIGHT_COLOR (Purple, palette 5)
    const SGR_DEFAULT_FG: &str = "\x1b[39m"; // Color::Default — "terminal foreground"

    /// A prompt whose color methods are set per-test. Rendering is inherited from
    /// `TestPrompt` since these tests only care about the emitted escapes.
    struct ColoredPrompt {
        prompt: Color,
        indicator: Color,
        right: Color,
    }

    impl Prompt for ColoredPrompt {
        fn render_prompt_left(&self) -> Cow<'_, str> {
            TestPrompt.render_prompt_left()
        }
        fn render_prompt_right(&self) -> Cow<'_, str> {
            TestPrompt.render_prompt_right()
        }
        fn render_prompt_indicator(&self, mode: PromptEditMode) -> Cow<'_, str> {
            TestPrompt.render_prompt_indicator(mode)
        }
        fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
            TestPrompt.render_prompt_multiline_indicator()
        }
        fn render_prompt_history_search_indicator(
            &self,
            search: PromptHistorySearch,
        ) -> Cow<'_, str> {
            TestPrompt.render_prompt_history_search_indicator(search)
        }

        fn get_prompt_color(&self) -> Color {
            self.prompt
        }
        fn get_indicator_color(&self) -> Color {
            self.indicator
        }
        fn get_prompt_right_color(&self) -> Color {
            self.right
        }
    }
    /// Like `capture_repaint`, but with configurable ANSI coloring. Also returns
    /// whether the paint took the large-buffer branch, since `repaint_buffer`
    /// recomputes that field and a caller cannot force it.
    fn capture_repaint_ansi(
        prompt: &dyn Prompt,
        lines: &PromptLines,
        use_ansi_coloring: bool,
    ) -> (String, bool) {
        let mut p = Painter::new(W::capture());
        p.terminal_size = (20, 10);
        p.prompt_start_row.mark_verified(0);
        p.relative_cursor_position = Some(Painter::relative_origin());
        p.prompt_height = 1;
        p.repaint_buffer(
            prompt,
            lines,
            PromptEditMode::Default,
            None,
            use_ansi_coloring,
            &None,
        )
        .expect("repaint_buffer failed");
        let large = p.large_buffer;
        (
            String::from_utf8_lossy(p.stdout.captured()).into_owned(),
            large,
        )
    }

    /// Records which palette entries crossterm's `SetForegroundColor` selected
    /// for the pre-migration defaults. crossterm's `Green`/`Cyan` are the
    /// *bright* entries (10/14) — `DarkGreen`/`DarkCyan` are 2/6 — so the
    /// nu-ansi-term replacements have to be the `Light*` variants to keep the
    /// prompt looking the same.
    ///
    /// Green and cyan intentionally re-encode from the 256-color form
    /// (`38;5;10`) to the aixterm form (`92`); both select palette entry 10.
    #[test]
    fn crossterm_defaults_were_the_bright_palette_entries() {
        use crossterm::{
            style::{Color as CtColor, SetForegroundColor},
            Command,
        };

        fn crossterm_sgr(color: CtColor) -> String {
            let mut buf = String::new();
            SetForegroundColor(color)
                .write_ansi(&mut buf)
                .expect("write_ansi failed");
            buf
        }

        for (name, crossterm, palette) in [
            ("prompt", CtColor::Green, 10),
            ("indicator", CtColor::Cyan, 14),
            ("right prompt", CtColor::AnsiValue(5), 5),
        ] {
            assert_eq!(
                crossterm_sgr(crossterm),
                Color::Fixed(palette).prefix().to_string(),
                "{name} default selected a different palette entry than assumed"
            );
        }

        // "Unstyled" has no palette entry; both spellings are SGR 39.
        assert_eq!(
            Color::Default.prefix().to_string(),
            crossterm_sgr(CtColor::Reset)
        );
    }

    /// The trait's default colors reach the terminal as the expected SGR
    /// sequences, on the small-buffer path.
    #[test]
    fn default_prompt_colors_emit_expected_sgr() {
        let (out, _) =
            capture_repaint_ansi(&TestPrompt, &make_lines("> ", "", "RP", "hi", ""), true);

        assert!(
            out.contains(SGR_GREEN),
            "left prompt color missing: {out:?}"
        );
        assert!(out.contains(SGR_CYAN), "indicator color missing: {out:?}");
        assert!(
            out.contains(SGR_PURPLE),
            "right prompt color missing: {out:?}"
        );
    }

    /// `Color::Default` must emit an explicit SGR 39, never an empty prefix —
    /// an empty one would let the active color bleed into an unstyled prompt,
    /// which is the starship bug (#1046).
    #[test]
    fn default_color_emits_explicit_foreground_reset() {
        let prompt = ColoredPrompt {
            prompt: Color::Default,
            indicator: Color::Default,
            right: Color::Default,
        };
        let (out, _) = capture_repaint_ansi(&prompt, &make_lines("> ", "", "RP", "hi", ""), true);

        assert!(
            out.contains(SGR_DEFAULT_FG),
            "Color::Default must emit an explicit SGR 39, not nothing: {out:?}"
        );
        assert!(
            !out.contains(SGR_GREEN),
            "no default color should leak through: {out:?}"
        );
    }

    /// `print_large_buffer` has its own three color call sites that the other
    /// capture tests never reach.
    ///
    /// The bulk sits in `after_cursor` so `extra_rows` stays 0 and the right
    /// prompt is still drawn; a tall `before_cursor` would suppress it (see
    /// `test_layout_right_prompt_suppressed_in_large_buffer`).
    #[test]
    fn large_buffer_path_emits_prompt_colors() {
        let tall = "line\n".repeat(15);
        let (out, large) =
            capture_repaint_ansi(&TestPrompt, &make_lines("> ", "", "RP", "hi", &tall), true);

        assert!(large, "expected the large-buffer path to be taken");
        assert!(
            out.contains(SGR_GREEN),
            "left prompt color missing: {out:?}"
        );
        assert!(out.contains(SGR_CYAN), "indicator color missing: {out:?}");
        assert!(
            out.contains(SGR_PURPLE),
            "right prompt color missing: {out:?}"
        );
    }

    /// Every color write stays inside its `use_ansi_coloring` guard. Checks the
    /// color sequences only — `repaint_buffer` always emits a leading `\x1b[0m`.
    #[test]
    fn no_prompt_colors_when_ansi_coloring_disabled() {
        let (out, _) =
            capture_repaint_ansi(&TestPrompt, &make_lines("> ", "", "RP", "hi", ""), false);

        for sgr in [SGR_GREEN, SGR_CYAN, SGR_PURPLE] {
            assert!(
                !out.contains(sgr),
                "emitted {sgr:?} with coloring disabled: {out:?}"
            );
        }
    }
}
