# Функциональные требования

## [FR-001] **Вход по PIN-коду** [BR-001]
Функциональное требование: система должнаallow users to set a 4-digit PIN during registration and use it for subsequent logins. PIN attempts limited to 5 before account lockout.

## [FR-002] **Биометрическая аутентификация** [BR-001]
Функциональное требование: система должна поддерживать вход через отпечаток пали (fingerprint) и распознавание лица (Face ID) на supported устройствах.

## [FR-003] **Просмотр баланса** [BR-002]
Функциональное требование: на главном экране после входа система должна отображать текущий баланс активного счета с обновлением в реальном времени.

## [FR-004] **История транзакций** [BR-002]
Функциональное требование: система должнаallow users to view transaction history за последние 90 дней с фильтрацией по дате, сумме и типу операции.

## [FR-005] **Перевод на свою карту** [BR-003]
Функциональное требование: система должнаallow users to transfer funds between their own accounts instantly without fees.

## [FR-006] **Перевод по номеру телефона** [BR-003]
Функциональное требование: система должнаallow users to send money to another person by entering their phone number, with confirmation via SMS code.

## [FR-007] **Перевод по реквизитам** [BR-003]
Функциональное требование: система должнаallow users to transfer money to a bank account using BIC and account number (IBAN).

## [FR-008] **Просмотр информации о карте** [BR-004]
Функциональное требование: система должна отображать информацию о карте: последние 4 цифры, срок действия, статус (активна/заблокирована).

## [FR-009] **Блокировка карты** [BR-004]
Функциональное требование: система должнаallow users to temporarily block their card with one tap, blocking all incoming transactions.

## [FR-010] **Разблокировка карты** [BR-004]
Функциональное требование: система должнаallow users to unblock their card after temporary block via PIN or biometric confirmation.

## [FR-011] **Настройка уведомлений** [BR-005]
Функциональное требование: система должнаallow users to configure which notifications they receive (transactions, security, promotions) via settings menu.

## [FR-012] **Уведомление о транзакции** [BR-005]
Функциональное требование: система должна отправлять push-уведомление immediately after each successful transaction with amount and merchant details.

## [FR-013] **Двухфакторная аутентификация** [BR-006]
Функциональное требование: система должна требовать подтверждение через SMS-код при входе с нового устройства.

## [FR-014] **Экспорт выписки** [BR-002]
Функциональное требование: система должнаallow users to export transaction history в формате PDF или CSV за выбранный период.
