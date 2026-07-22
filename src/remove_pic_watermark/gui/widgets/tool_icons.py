"""Simple vector icons for toolbar / tool rail (no external asset pack)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def _px(size: int = 20) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    return pm


def _painter(pm: QPixmap) -> QPainter:
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return p


def _stroke(color: str = "#344054", width: float = 1.6) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def icon_open(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    # folder
    p.drawRoundedRect(QRectF(3, 7, 14, 10), 1.5, 1.5)
    p.drawLine(QPointF(3, 9), QPointF(8, 9))
    p.drawLine(QPointF(8, 9), QPointF(10, 7))
    p.drawLine(QPointF(10, 7), QPointF(17, 7))
    p.end()
    return QIcon(pm)


def icon_save(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.5, 3, 13, 14), 1.5, 1.5)
    p.drawRect(QRectF(6.5, 3, 7, 5))
    p.drawLine(QPointF(6.5, 12), QPointF(13.5, 12))
    p.drawLine(QPointF(6.5, 14.5), QPointF(13.5, 14.5))
    p.end()
    return QIcon(pm)


def icon_undo(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    pen = _stroke()
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    # curved arrow left
    p.drawArc(QRectF(4, 4, 12, 12), 40 * 16, 200 * 16)
    p.setBrush(QColor("#344054"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(
        [
            QPointF(4.5, 5.5),
            QPointF(9.5, 4.0),
            QPointF(8.2, 8.8),
        ]
    )
    p.end()
    return QIcon(pm)


def icon_redo(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(4, 4, 12, 12), -40 * 16, -200 * 16)
    p.setBrush(QColor("#344054"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(
        [
            QPointF(15.5, 5.5),
            QPointF(10.5, 4.0),
            QPointF(11.8, 8.8),
        ]
    )
    p.end()
    return QIcon(pm)


def icon_zoom_in(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(8.5, 8.5), 5.2, 5.2)
    p.drawLine(QPointF(12.2, 12.2), QPointF(16.5, 16.5))
    p.drawLine(QPointF(6.2, 8.5), QPointF(10.8, 8.5))
    p.drawLine(QPointF(8.5, 6.2), QPointF(8.5, 10.8))
    p.end()
    return QIcon(pm)


def icon_zoom_out(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(8.5, 8.5), 5.2, 5.2)
    p.drawLine(QPointF(12.2, 12.2), QPointF(16.5, 16.5))
    p.drawLine(QPointF(6.2, 8.5), QPointF(10.8, 8.5))
    p.end()
    return QIcon(pm)


def icon_zoom_fit(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    # four corner brackets
    for x0, y0, dx, dy in (
        (3, 3, 1, 1),
        (17, 3, -1, 1),
        (3, 17, 1, -1),
        (17, 17, -1, -1),
    ):
        p.drawLine(QPointF(x0, y0), QPointF(x0 + dx * 4, y0))
        p.drawLine(QPointF(x0, y0), QPointF(x0, y0 + dy * 4))
    p.drawRect(QRectF(6.5, 6.5, 7, 7))
    p.end()
    return QIcon(pm)


def icon_zoom_1x(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke("#344054", 1.4))
    p.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "1:1")
    p.end()
    return QIcon(pm)


def icon_clear(size: int = 20) -> QIcon:
    """Clear current selection / paint strokes (main canvas tool — not queue empty)."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke("#b42318", 1.7))
    p.drawEllipse(QPointF(10, 10), 6.5, 6.5)
    p.drawLine(QPointF(6.2, 6.2), QPointF(13.8, 13.8))
    p.end()
    return QIcon(pm)


def icon_empty_queue(size: int = 20) -> QIcon:
    """Empty list / queue / dataset — distinct from canvas clear (icon_clear).

    Visual: three list rows + small corner slash (session reset, not stroke erase).
    """
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke("#344054", 1.5))
    # list rows
    for y in (5.5, 10.0, 14.5):
        p.drawLine(QPointF(3.5, y), QPointF(12.5, y))
    # badge slash (not full forbidden circle — different silhouette)
    p.setPen(_stroke("#b42318", 1.6))
    p.drawLine(QPointF(11.5, 4.5), QPointF(16.5, 15.5))
    p.drawLine(QPointF(16.5, 4.5), QPointF(11.5, 15.5))
    p.end()
    return QIcon(pm)


def icon_rect(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor(37, 99, 235, 35))
    p.drawRect(QRectF(4, 5, 12, 10))
    p.end()
    return QIcon(pm)


def icon_brush(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor("#344054"))
    # tip
    p.drawPolygon(
        [
            QPointF(5, 15),
            QPointF(7.5, 12.5),
            QPointF(9.5, 14.5),
        ]
    )
    pen = _stroke()
    pen.setWidthF(2.2)
    p.setPen(pen)
    p.drawLine(QPointF(8.5, 13.5), QPointF(15, 5.5))
    p.end()
    return QIcon(pm)


def icon_eraser(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor("#e4e7ec"))
    p.translate(10, 10)
    p.rotate(-35)
    p.drawRoundedRect(QRectF(-6, -3.5, 12, 7), 1.5, 1.5)
    p.setBrush(QColor("#f97066"))
    p.drawRect(QRectF(-6, -3.5, 5, 7))
    p.end()
    return QIcon(pm)


def icon_pan(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    pen = _stroke()
    pen.setWidthF(1.5)
    p.setPen(pen)
    # cross arrows
    p.drawLine(QPointF(10, 3.5), QPointF(10, 16.5))
    p.drawLine(QPointF(3.5, 10), QPointF(16.5, 10))
    p.setBrush(QColor("#344054"))
    p.setPen(Qt.PenStyle.NoPen)
    for pts in (
        [QPointF(10, 3), QPointF(7.8, 6.2), QPointF(12.2, 6.2)],
        [QPointF(10, 17), QPointF(7.8, 13.8), QPointF(12.2, 13.8)],
        [QPointF(3, 10), QPointF(6.2, 7.8), QPointF(6.2, 12.2)],
        [QPointF(17, 10), QPointF(13.8, 7.8), QPointF(13.8, 12.2)],
    ):
        p.drawPolygon(pts)
    p.end()
    return QIcon(pm)


def icon_play(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1570ef"))
    p.drawPolygon(
        [
            QPointF(6, 4),
            QPointF(16, 10),
            QPointF(6, 16),
        ]
    )
    p.end()
    return QIcon(pm)


def icon_stop(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#d92d20"))
    p.drawRoundedRect(QRectF(5.5, 5.5, 9, 9), 1.5, 1.5)
    p.end()
    return QIcon(pm)


def icon_reload(size: int = 20) -> QIcon:
    """Circular arrow — restore / reload original."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(4, 4, 12, 12), 50 * 16, 260 * 16)
    p.setBrush(QColor("#344054"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(
        [
            QPointF(14.5, 4.5),
            QPointF(17.2, 8.2),
            QPointF(12.0, 8.5),
        ]
    )
    p.end()
    return QIcon(pm)


def icon_folder(size: int = 20) -> QIcon:
    return icon_open(size)


def icon_add(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.drawEllipse(QPointF(10, 10), 7, 7)
    p.drawLine(QPointF(10, 6.5), QPointF(10, 13.5))
    p.drawLine(QPointF(6.5, 10), QPointF(13.5, 10))
    p.end()
    return QIcon(pm)


def icon_remove(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke("#b42318"))
    p.drawEllipse(QPointF(10, 10), 7, 7)
    p.drawLine(QPointF(6.5, 10), QPointF(13.5, 10))
    p.end()
    return QIcon(pm)


def icon_refresh(size: int = 20) -> QIcon:
    return icon_reload(size)


def icon_folder_open(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3, 6, 14, 11), 1.5, 1.5)
    p.drawLine(QPointF(3, 9), QPointF(17, 9))
    p.end()
    return QIcon(pm)


def icon_mask(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor(15, 23, 42))
    p.drawRoundedRect(QRectF(3, 4, 14, 12), 2, 2)
    p.setBrush(QColor(248, 250, 252))
    p.drawEllipse(QPointF(10, 10), 3.5, 3.5)
    p.end()
    return QIcon(pm)


def icon_debug(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor(239, 68, 68, 80))
    p.drawRect(QRectF(4, 5, 12, 10))
    p.setPen(_stroke("#ef4444", 1.4))
    p.drawLine(QPointF(6, 8), QPointF(14, 14))
    p.end()
    return QIcon(pm)


def icon_delete(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke("#b42318"))
    p.drawLine(QPointF(6, 7), QPointF(14, 7))
    p.drawLine(QPointF(8, 7), QPointF(8.5, 5))
    p.drawLine(QPointF(12, 7), QPointF(11.5, 5))
    p.drawLine(QPointF(8.5, 5), QPointF(11.5, 5))
    p.drawRect(QRectF(6.5, 7, 7, 9))
    p.drawLine(QPointF(9, 9), QPointF(9, 14))
    p.drawLine(QPointF(11, 9), QPointF(11, 14))
    p.end()
    return QIcon(pm)


def icon_ai(size: int = 20) -> QIcon:
    """Spark / AI badge."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#7c3aed"))
    # four-point star
    pts = [
        QPointF(10, 2.5),
        QPointF(11.6, 7.8),
        QPointF(17.2, 8.2),
        QPointF(12.8, 11.4),
        QPointF(14.4, 16.8),
        QPointF(10, 13.6),
        QPointF(5.6, 16.8),
        QPointF(7.2, 11.4),
        QPointF(2.8, 8.2),
        QPointF(8.4, 7.8),
    ]
    p.drawPolygon(pts)
    p.end()
    return QIcon(pm)


def icon_back(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(14, 10), QPointF(5, 10))
    p.drawLine(QPointF(8.5, 6.5), QPointF(5, 10))
    p.drawLine(QPointF(8.5, 13.5), QPointF(5, 10))
    p.end()
    return QIcon(pm)


def icon_device(size: int = 20) -> QIcon:
    """GPU / device chip icon (simple chip + pins)."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(QColor(37, 99, 235, 40))
    p.drawRoundedRect(QRectF(4, 6, 12, 8), 1.5, 1.5)
    p.setPen(_stroke("#1570ef", 1.3))
    for x in (6.5, 10, 13.5):
        p.drawLine(QPointF(x, 6), QPointF(x, 4.2))
        p.drawLine(QPointF(x, 14), QPointF(x, 15.8))
    p.end()
    return QIcon(pm)


def icon_preview(size: int = 20) -> QIcon:
    """Eye / preview."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 7, 4.5)
    p.setBrush(QColor("#344054"))
    p.drawEllipse(QPointF(10, 10), 2.2, 2.2)
    p.end()
    return QIcon(pm)


def icon_prev(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(13, 5), QPointF(7, 10))
    p.drawLine(QPointF(7, 10), QPointF(13, 15))
    p.end()
    return QIcon(pm)


def icon_next(size: int = 20) -> QIcon:
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(7, 5), QPointF(13, 10))
    p.drawLine(QPointF(13, 10), QPointF(7, 15))
    p.end()
    return QIcon(pm)


def icon_grid(size: int = 20) -> QIcon:
    """Layout / grid."""
    pm = _px(size)
    p = _painter(pm)
    p.setPen(_stroke())
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.5, 3.5, 13, 13))
    p.drawLine(QPointF(10, 3.5), QPointF(10, 16.5))
    p.drawLine(QPointF(3.5, 10), QPointF(16.5, 10))
    p.end()
    return QIcon(pm)
